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
    # Video settings (paper-aligned)
    # -----------------------------
    record_video = bool(args.get("record_video", False))
    video_every = int(args.get("video_every", 1))          # registra 1 episodio ogni N
    video_fps = int(args.get("video_fps", 2))
    video_size = args.get("video_size", (464, 464))
    record_n_episodes = int(args.get("record_n_episodes", 1))

    def should_record_episode(ep_idx_1based: int) -> bool:
        return record_video and (ep_idx_1based % video_every == 0)

    # ------------------------------------------------------------------
    # Enforce 1-step episodes (paper): one chemical edit per counterfactual
    # ------------------------------------------------------------------
    if record_video:
        # forziamo 1 step per episodio per evitare "crescita" multi-step
        args["max_steps_per_episode"] = 1
        environment.max_steps = 1  # sincronizza l'env MolDQN/Molecule

    # Lista frame globale: seed + 1 frame per episodio registrato
    frames_all = []
    recorded_episodes = 0

    def record_current_frame(info: dict):
        """Renderizza lo stato corrente (molecola corrente) come frame video."""
        frame = environment.render(mode="rgb_array", info=info, size=video_size)
        frame = np.require(frame, dtype=np.uint8, requirements=["C_CONTIGUOUS", "ALIGNED"])
        frames_all.append(frame)

    # Registra subito la seed (molecola iniziale)
    if record_video:
        record_current_frame({
            "type": "seed",
            "episode": 0,
            "step": 0,
            "action": "INIT",
            "reward": "0.0000",
            "pred": "0.0000",
            "sim": "1.0000",
            "eps": f"{1.0:.3f}",
        })

    # -----------------------------
    # Training loop
    # -----------------------------
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

            # decaying epsilon
            eps *= 0.9987

            # ----------------------------------------------------------
            # Video: 1 frame per episodio = molecola dopo 1 sola modifica
            # ----------------------------------------------------------
            if should_record_episode(episode):
                record_current_frame({
                    "type": "cf",
                    "episode": episode,
                    "step": environment.num_steps_taken,  # sarà 1
                    "action": str(action),
                    "reward": f"{out['reward']:.4f}",
                    "pred": f"{out['reward_pred']:.4f}",
                    "sim": f"{out['reward_sim']:.4f}",
                    "eps": f"{eps:.3f}",
                })
                recorded_episodes += 1

            # reset episodio: torna alla seed
            environment.initialize()

            # ----------------------------------------------------------
            # Stop e salva UN SOLO video con: seed + N controfattuali
            # ----------------------------------------------------------
            if record_video and (recorded_episodes >= record_n_episodes):
                video_dir = os.path.join(base_path, "videos")
                os.makedirs(video_dir, exist_ok=True)

                mp4_path = os.path.join(
                    video_dir,
                    f"sample_{args['sample']}_seed_plus_{recorded_episodes}_cf.mp4"
                )

                try:
                    frames_safe = [
                        np.require(f, dtype=np.uint8, requirements=["C_CONTIGUOUS", "ALIGNED"])
                        for f in frames_all
                    ]
                    imageio.mimsave(mp4_path, frames_safe, fps=video_fps)
                    print("Saved rollout video:", mp4_path)
                except Exception as e:
                    gif_path = mp4_path.replace(".mp4", ".gif")
                    imageio.mimsave(gif_path, frames_all, fps=video_fps)
                    print("MP4 failed, saved GIF instead:", gif_path, "| error:", str(e))

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
    video_size: int = typer.Option(448),         # user-friendly: un solo int
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

def compute_cf_metrics_tox21(original_entry: dict, cf_list: list, sim_thresholds=(0.9, 0.8, 0.7)):
    """
    original_entry: dict come quello che aggiungi in overall_queue[0] (marker='og', prediction, ...)
    cf_list: lista di dict (cf_queue.data_) con campi: reward, reward_pred, reward_sim, prediction{class, output}, ...
    """
    og_class = int(original_entry["prediction"]["class"])

    # safety: filtra solo CF con prediction
    cfs = [c for c in cf_list if "prediction" in c and "class" in c["prediction"]]

    if len(cfs) == 0:
        return {
            "n_cf": 0,
            "flip_rate": 0.0,
            "tox_to_nontox_rate": 0.0,
            "nontox_to_tox_rate": 0.0,
        }

    cf_class = np.array([int(c["prediction"]["class"]) for c in cfs], dtype=int)
    flips = (cf_class != og_class).astype(np.int32)

    # similarity: nel vostro codice è già out['reward_sim']
    sims = np.array([float(c.get("reward_sim", np.nan)) for c in cfs], dtype=float)
    # reward_pred: nel vostro codice è out['reward_pred']
    pred_scores = np.array([float(c.get("reward_pred", np.nan)) for c in cfs], dtype=float)
    rewards = np.array([float(c.get("reward", np.nan)) for c in cfs], dtype=float)

    flip_rate = float(flips.mean())

    # Direzioni (assumendo classi: 1=Tox, 0=NoTox; se è invertito nel tuo setup, basta invertire qui)
    tox_to_nontox_rate = float(np.mean((og_class == 1) & (cf_class == 0))) if og_class == 1 else 0.0
    nontox_to_tox_rate = float(np.mean((og_class == 0) & (cf_class == 1))) if og_class == 0 else 0.0

    # Success@k: esiste almeno una CF che flippa
    success_at_k = float(flips.max())  # 1.0 se almeno un flip

    # Similarity sui soli flip (se ce ne sono)
    flip_sims = sims[flips == 1]
    sim_flip_median = float(np.nanmedian(flip_sims)) if flip_sims.size else float("nan")
    sim_all_median = float(np.nanmedian(sims))

    # Flip rate condizionato a soglie di similarity
    flip_rate_at_sim = {}
    for t in sim_thresholds:
        mask = sims >= t
        flip_rate_at_sim[f"flip_rate_sim_ge_{t}"] = float(flips[mask].mean()) if mask.any() else 0.0

    return {
        "n_cf": int(len(cfs)),
        "flip_rate": flip_rate,
        "tox_to_nontox_rate": tox_to_nontox_rate,
        "nontox_to_tox_rate": nontox_to_tox_rate,
        "success_at_k": success_at_k,
        "sim_all_median": sim_all_median,
        "sim_flip_median": sim_flip_median,
        "pred_score_flip_mean": float(np.nanmean(pred_scores[flips == 1])) if (flips == 1).any() else float("nan"),
        "reward_flip_mean": float(np.nanmean(rewards[flips == 1])) if (flips == 1).any() else float("nan"),
        **flip_rate_at_sim,
        # per TensorBoard histogram
        "_sims": sims,
        "_flip_sims": flip_sims,
        "_flips": flips,
    }


if __name__ == "__main__":
    typer.run(main)
