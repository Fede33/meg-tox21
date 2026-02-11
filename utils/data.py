import os
import torch
import random

from torch_geometric.data import DataLoader
from torch.nn import functional as F
from torch_geometric.datasets import TUDataset

from utils.molecules import check_molecule_validity, pyg_to_mol_tox21


def pre_transform(sample, n_pad: int):
    sample.x = F.pad(sample.x, (0, n_pad), "constant")
    return sample


def get_split(dataset_name: str, split: str, experiment: str):
    """
    Load a previously saved split from runs/tox21/<experiment>/splits/<split>.pth
    and return it as a TUDataset object with proper (data, slices).
    """
    if dataset_name.lower() != 'tox21':
        raise ValueError(f"Only 'tox21' is supported. Got: {dataset_name}")

    ds = TUDataset(
        'data/tox21',
        name='Tox21_AhR_testing',
        pre_transform=lambda sample: pre_transform(sample, 2)
    )

    ds.data, ds.slices = torch.load(f"runs/tox21/{experiment}/splits/{split}.pth")
    return ds


def preprocess(dataset_name: str, experiment_name: str, batch_size: int):
    """
    Main entrypoint called by train_dgn.py.
    Returns:
      train_loader, val_loader, test_loader, train_ds, val_ds, test_ds, num_features, num_classes
    """
    if dataset_name.lower() != 'tox21':
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. This project supports only 'tox21'."
        )

    return _preprocess_tox21(experiment_name, batch_size)


def _preprocess_tox21(experiment_name: str, batch_size: int):
    dataset_tr = TUDataset(
        'data/tox21',
        name='Tox21_AhR_training',
        pre_transform=lambda sample: pre_transform(sample, 3)
    )

    dataset_vl = TUDataset(
        'data/tox21',
        name='Tox21_AhR_evaluation',
        pre_transform=lambda sample: pre_transform(sample, 0)
    )

    dataset_ts = TUDataset(
        'data/tox21',
        name='Tox21_AhR_testing',
        pre_transform=lambda sample: pre_transform(sample, 2)
    )


    data_list = (
        [dataset_tr.get(idx) for idx in range(len(dataset_tr))] +
        [dataset_vl.get(idx) for idx in range(len(dataset_vl))] +
        [dataset_ts.get(idx) for idx in range(len(dataset_ts))]
    )

    data_list = list(filter(lambda mol: check_molecule_validity(mol, pyg_to_mol_tox21), data_list))

    
    positives = [x for x in data_list if int(x.y) == 1]
    negatives = [x for x in data_list if int(x.y) == 0]
    n_pos = len(positives)
    negatives = negatives[:n_pos]

    data_list = positives + negatives
    random.shuffle(data_list)

    
    n = len(data_list) // 10
    train_data = data_list[n:]
    val_data = data_list[:n]
    test_data = train_data[:n]
    train_data = train_data[n:]

    train = dataset_tr
    val = dataset_vl
    test = dataset_ts

    train.data, train.slices = train.collate(train_data)
    val.data, val.slices = train.collate(val_data)
    test.data, test.slices = train.collate(test_data)

    
    base = f"runs/tox21/{experiment_name}/splits"
    os.makedirs(base, exist_ok=True)
    torch.save((train.data, train.slices), f"{base}/train.pth")
    torch.save((val.data, val.slices), f"{base}/val.pth")
    torch.save((test.data, test.slices), f"{base}/test.pth")

    return (
        DataLoader(train, batch_size=batch_size),
        DataLoader(val,   batch_size=batch_size),
        DataLoader(test,  batch_size=batch_size),
        train,
        val,
        test,
        max(train.num_features, val.num_features, test.num_features),
        train.num_classes,
    )