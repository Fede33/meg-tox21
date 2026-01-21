import os
import json
import random

import imageio.v2 as imageio

import numpy as np
import torch
import typer
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from rdkit import Chem
from rdkit.Chem import Draw

from models.explainer import CF_Tox21, Agent
from utils import (
    SortedQueue,
    morgan_bit_fingerprint,
    get_split,
    get_dgn,
    mol_to_smiles,
    x_map_tox21,
    pyg_to_mol_tox21,
)

# -----------------------------
# TOX21 pipeline
# -----------------------------
def tox21(
    general_params,
    base_path,
    writer,
    num_counterfactuals,
    original_molecule,
    model_to_explain,
    **args,
):
    # Forward pass on original molecule
    out, (_, original_encoding) = model_to_explain(
        original_molecule.x,
        original_molecule.edge_index
    )

    logits = F.softmax(out, dim=-1).detach().squeeze()
    pred_class = logits.argmax().item()

    # Safety check: in questo progetto spiegate predizioni corrette
    assert pred_class == original_molecule.y.item()

    # Compute SMILES from pyg molecule
    original_molecule.smiles = mol_to_smiles(pyg_to_mol_tox21(original_molecule))
    print(f"Molecule: {original_molecule.smiles}")

    # Infer atom types present in the molecule (for action space restriction)
    atoms_ = [
        x_map_tox21(e).name
        for e in np.unique(
            [x.tolist().index(1) for x in original_molecule.x.numpy()]
        )
    ]

    params = {
        # General-purpose params
        **general_params,
        "init_mol": original_molecule.smiles,
        "atom_types": set(atoms_),

        # Task-specific params
        "original_molecule": original_molecule,
        "model_to_explain": model_to_explain,
        "weight_sim": 0.2,
        "similarity_measure": "combined",
    }

    cf_queue = SortedQueue(
        num_counterfactuals,
        sort_predicate=lambda mol: mol["reward"]
    )

    cf_env = CF_Tox21(**params)
    cf_env.initialize()

    def action_encoder(action_smiles: str):
        return morgan_bit_fingerprint(
            action_smiles,
            args["fp_length"],
            args["fp_radius"]
        ).numpy()

    meg_train(
        writer=writer,
        action_encoder=action_encoder,
        n_input=args["fp_length"],
        environment=cf_env,
        queue=cf_queue,
        marker="cf",
        tb_name="tox21",
        id_function=lambda action: action,
        args=args,
        base_path=base_path,
    )

    overall_queue = [
        {
            "pyg": original_molecule,
            "marker": "og",
            "smiles": original_molecule.smiles,
            "encoding": original_encoding.numpy(),
            "prediction": {
                "type": "bin_classification",
                "output": logits.numpy().tolist(),
                "for_explanation": original_molecule.y.item(),
                "class": original_molecule.y.item(),
            },
        }
    ]
    overall_queue.extend(cf_queue.data_)

    save_results(base_path, overall_queue, args)


# -----------------------------
# MEG training loop (DQN)
# -----------------------------
def meg_train(
    writer,
    action_encoder,
    n_input,
    environment,
    queue,
    marker,
    tb_name,
    id_function,
    args,
    base_path,
):
    device = torch.device("cpu")
    agent = Agent(n_input + 1, 1, device, args["lr"], args["replay_buffer_size"])

    eps = 1.0
    episode = 0
    it = 0
    
    # -----------------------------
    # Video settings (safe defaults)
    # -----------------------------
    record_video = args.get("record_video", True)
    video_every = int(args.get("video_every", 1))          # registra 1 episodio ogni N
    video_fps = int(args.get("video_fps", 2))
    video_size = args.get("video_size", (448, 448))
    record_n_episodes = int(args.get("record_n_episodes", 1))

    frames = []
    recorded_episodes = 0

    def should_record_episode(ep_idx_1based: int) -> bool:
        return record_video and (ep_idx_1based % video_every == 0)

    while episode < args["epochs"]:
        steps_left = args["max_steps_per_episode"] - environment.num_steps_taken
        valid_actions = list(environment.get_valid_actions())

        observations = np.vstack(
            [np.append(action_encoder(a), steps_left) for a in valid_actions]
        )
        observations = torch.as_tensor(observations).float()

        a_idx = agent.action_step(observations, eps)
        action = valid_actions[a_idx]

        result = environment.step(action)
        action_embedding = np.append(action_encoder(action), steps_left)

        _, out, done = result

        # -------------
        # Render frame
        # -------------
        if should_record_episode(episode + 1):
            info = {
                "episode": episode + 1,
                "step": environment.num_steps_taken,
                "steps_left": steps_left,
                "action": str(action),
                "reward": f"{out['reward']:.4f}",
                "pred": f"{out['reward_pred']:.4f}",
                "sim": f"{out['reward_sim']:.4f}",
                "eps": f"{eps:.3f}",
            }

            frame = environment.render(mode="rgb_array", info=info, size=video_size)
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            frames.append(frame)


        writer.add_scalar(f"{tb_name}/reward", out["reward"], it)
        writer.add_scalar(f"{tb_name}/prediction", out["reward_pred"], it)
        writer.add_scalar(f"{tb_name}/similarity", out["reward_sim"], it)

        steps_left = args["max_steps_per_episode"] - environment.num_steps_taken
        next_action_embeddings = np.vstack(
            [np.append(action_encoder(a), steps_left) for a in environment.get_valid_actions()]
        )

        agent.replay_buffer.push(
            torch.as_tensor(action_embedding).float(),
            torch.as_tensor(out["reward"]).float(),
            torch.as_tensor(next_action_embeddings).float(),
            float(result.terminated),
        )

        if (
            it % args["update_interval"] == 0
            and len(agent.replay_buffer) >= args["batch_size"]
        ):
            _ = agent.train_step(args["batch_size"], args["gamma"], args["polyak"])

        it += 1

        if done:
            episode += 1
            print(
                f'({args["sample"]}) Episode {episode}> '
                f'Reward = {out["reward"]:.4f} '
                f'(pred: {out["reward_pred"]:.4f}, sim: {out["reward_sim"]:.4f})'
            )

            queue.insert({"marker": marker, "id": id_function(action), **out})

            # decaying epsilon (come nel paper/repo)
            eps *= 0.9987

            # -----------------------------
            # Save video at end of episode
            # -----------------------------
            if should_record_episode(episode) and len(frames) > 0:
                video_dir = os.path.join(base_path, "videos")
                os.makedirs(video_dir, exist_ok=True)

                mp4_path = os.path.join(
                    video_dir, f"sample_{args['sample']}_ep_{episode}.mp4"
                )

                try:
                    imageio.mimsave(mp4_path, frames, fps=video_fps)
                    print("Saved rollout video:", mp4_path)
                except Exception as e:
                    gif_path = mp4_path.replace(".mp4", ".gif")
                    imageio.mimsave(gif_path, frames, fps=video_fps)
                    print("MP4 failed, saved GIF instead:", gif_path, "| error:", str(e))

                recorded_episodes += 1

            # reset per prossimo episodio
            frames = []
            environment.initialize()

            # stop dopo aver registrato N episodi (per demo)
            if record_video and (recorded_episodes >= record_n_episodes):
                print(f"Stopping: recorded {recorded_episodes} episode(s).")
                return



# -----------------------------
# Save outputs
# -----------------------------
def save_results(base_path, queue, args):
    output_dir = os.path.join(base_path, "meg_output", str(args["sample"]))
    embedding_dir = os.path.join(output_dir, "embeddings")

    os.makedirs(embedding_dir, exist_ok=True)

    for i, molecule in enumerate(queue):
        np.save(os.path.join(embedding_dir, f"{i}"), molecule.pop("encoding"))
        _ = molecule.pop("pyg")  # remove non-serializable

    with open(os.path.join(output_dir, "seed"), "w") as outf:
        json.dump(args["seed"], outf)

    with open(os.path.join(output_dir, "data.json"), "w") as outf:
        json.dump(queue, outf, indent=2)

def pick_correct_sample(dataset, experiment_name, split, model, start_idx=0, max_tries=500):
    data_list = get_split(dataset, split, experiment_name)
    model.eval()

    with torch.no_grad():
        for i in range(start_idx, min(len(data_list), start_idx + max_tries)):
            mol = data_list[i]
            out, _ = model(mol.x, mol.edge_index)
            pred = F.softmax(out, dim=-1).argmax(dim=-1).item()
            y = mol.y.item()

            if pred == y:
                print(f"[OK] using sample {i} (pred={pred}, y={y})")
                return mol, i

    raise RuntimeError("No correctly-predicted sample found.")


# -----------------------------
# CLI (TOX21 only)
# -----------------------------
def main(
    experiment_name: str = typer.Argument("test"),
    sample: int = typer.Option(0),
    epochs: int = typer.Option(5000),
    max_steps_per_episode: int = typer.Option(1),
    num_counterfactuals: int = typer.Option(10),
    fp_length: int = typer.Option(1024),
    fp_radius: int = typer.Option(2),
    lr: float = typer.Option(1e-4),
    polyak: float = typer.Option(0.995),
    gamma: float = typer.Option(0.95),
    discount: float = typer.Option(0.9),
    replay_buffer_size: int = typer.Option(10000),
    batch_size: int = typer.Option(1),
    update_interval: int = typer.Option(1),
    allow_no_modification: bool = typer.Option(False),
    allow_removal: bool = typer.Option(True),
    allow_node_addition: bool = typer.Option(True),
    allow_edge_addition: bool = typer.Option(True),
    allow_bonds_between_rings: bool = typer.Option(True),
    seed: int = typer.Option(random.randint(0, 2**12)),
    record_video: bool = typer.Option(True),
    video_every: int = typer.Option(1),          # 1 = registra OGNI episodio (per demo)
    video_fps: int = typer.Option(2),
    video_size: int = typer.Option(450),         # user-friendly: un solo int
    record_n_episodes: int = typer.Option(1),    # quanti episodi vuoi registrare (demo=1)
):
    dataset = "tox21"

    general_params = {
        "discount_factor": discount,
        "allow_removal": allow_removal,
        "allow_no_modification": allow_no_modification,
        "allow_bonds_between_rings": allow_bonds_between_rings,
        "allow_node_addition": allow_node_addition,
        "allow_edge_addition": allow_edge_addition,
        "allowed_ring_sizes": set([5, 6]),
        "max_steps": max_steps_per_episode,
        "fp_len": fp_length,
        "fp_rad": fp_radius,
    }

    torch.manual_seed(seed)

    base_path = f"./runs/{dataset}/{experiment_name}"

    model = get_dgn(dataset, experiment_name)
    original_molecule, sample = pick_correct_sample(
        dataset=dataset,
        experiment_name=experiment_name,
        split="test",
        model=model,
        start_idx=sample
    )
    tox21(
        general_params=general_params,
        base_path=base_path,
        writer=SummaryWriter(f"{base_path}/plots"),
        num_counterfactuals=num_counterfactuals,
        original_molecule=original_molecule,
        model_to_explain=model,
        experiment_name=experiment_name,
        sample=sample,
        epochs=epochs,
        max_steps_per_episode=max_steps_per_episode,
        fp_length=fp_length,
        fp_radius=fp_radius,
        lr=lr,
        polyak=polyak,
        gamma=gamma,
        discount=discount,
        replay_buffer_size=replay_buffer_size,
        batch_size=batch_size,
        update_interval=update_interval,
        seed=seed,
        record_video=record_video,
        video_every=video_every,
        video_fps=video_fps,
        video_size=(video_size, video_size),
        record_n_episodes=record_n_episodes,
    )


if __name__ == "__main__":
    typer.run(main)
