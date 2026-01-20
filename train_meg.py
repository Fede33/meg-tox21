import os
import json
import random

import numpy as np
import torch
import typer
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

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
):
    device = torch.device("cpu")
    agent = Agent(n_input + 1, 1, device, args["lr"], args["replay_buffer_size"])

    eps = 1.0
    episode = 0
    it = 0

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

            # decaying epsilon (come nel paper/repo)
            eps *= 0.9987

            environment.initialize()


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

    tox21(
        general_params=general_params,
        base_path=base_path,
        writer=SummaryWriter(f"{base_path}/plots"),
        num_counterfactuals=num_counterfactuals,
        original_molecule=get_split(dataset, "test", experiment_name)[sample],
        model_to_explain=get_dgn(dataset, experiment_name),
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
    )


if __name__ == "__main__":
    typer.run(main)
