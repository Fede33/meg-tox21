import os
import os.path as osp
import json

import torch
import typer

from models.encoder import GCNN
from utils import preprocess, train_cycle_classifier


def main(
    dataset_name: str,
    experiment_name: str = typer.Argument("test"),
    lr: float = typer.Option(0.01),
    hidden_size: int = typer.Option(32),
    batch_size: int = typer.Option(32),
    dropout: float = typer.Option(0.1),
    epochs: int = typer.Option(50),
    seed: int = typer.Option(0),
):
    dataset_name = dataset_name.lower()
    if dataset_name != "tox21":
        raise typer.BadParameter(
            f"Only 'tox21' is supported in this cleaned version. Got: '{dataset_name}'."
        )

    torch.manual_seed(seed)

    base_path = osp.join("./runs", dataset_name, experiment_name)

    # Create / reset folders
    if not osp.exists(base_path):
        os.makedirs(osp.join(base_path, "ckpt"), exist_ok=True)
        os.makedirs(osp.join(base_path, "plots"), exist_ok=True)
        os.makedirs(osp.join(base_path, "splits"), exist_ok=True)
        os.makedirs(osp.join(base_path, "meg_output"), exist_ok=True)
    else:
        import shutil
        shutil.rmtree(osp.join(base_path, "plots"), ignore_errors=True)
        os.makedirs(osp.join(base_path, "plots"), exist_ok=True)

    train_loader, val_loader, test_loader, *extra = preprocess(
        dataset_name, experiment_name, batch_size
    )
    train_ds, val_ds, test_ds, num_features, num_classes = extra

    len_train = len(train_ds)
    len_val = len(val_ds)
    len_test = len(test_ds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GCNN(
        num_input=num_features,
        num_hidden=hidden_size,
        num_output=num_classes,
        dropout=dropout,
    ).to(device)

    with open(osp.join(base_path, "hyperparams.json"), "w") as outf:
        json.dump(
            {
                "num_input": num_features,
                "num_hidden": hidden_size,
                "num_output": num_classes,
                "dropout": dropout,
                "seed": seed,
                "lr": lr,
                "batch_size": batch_size,
                "epochs": epochs,
            },
            outf,
            indent=2,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # TOX21 is a classification task here
    train_cycle_classifier(
        task=dataset_name,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        len_train=len_train,
        len_val=len_val,
        len_test=len_test,
        model=model,
        optimizer=optimizer,
        device=device,
        base_path=base_path,
        epochs=epochs,
    )


if __name__ == "__main__":
    typer.run(main)
