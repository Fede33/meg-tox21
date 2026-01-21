# MEG: Molecular Explanation Generator (TOX21)

This repository contains an implementation of **MEG – Molecular Explanation Generator**, an explainability framework for molecular graph neural networks, **based on the paper**:

> **MEG: Generating Molecular Counterfactual Explanations for Deep Graph Networks**  
> Danilo Numeroso, Davide Bacciu  
> *International Joint Conference on Neural Networks (IJCNN), 2021*  
> https://arxiv.org/abs/2104.08060

This version of work has been **adapted and simplified** to focus **exclusively on the TOX21 toxicity prediction task**.

---

## Project Overview

Modern Graph Neural Networks (GNNs) achieve strong performance on molecular property prediction tasks, but their decisions are often hard to interpret.

MEG addresses this problem by generating **molecular counterfactual explanations**:
- molecules that are **chemically valid**
- **similar** to the original molecule
- but that lead to a **different model prediction**

This implementation uses:
- a **Deep Graph Network (DGN)** trained on **TOX21**
- a **reinforcement learning agent (DQN)** to explore valid molecular edits
- a **multi-objective reward** combining:
  - prediction contrast
  - structural similarity (Morgan fingerprints + latent space similarity)

---

## Repository Structure (DA MODIFICARE POI ALLA FINE)

```text
MEG/
├── install.sh              # Environment setup script
├── train_dgn.py            # Train the GNN on TOX21
├── train_meg.py            # Generate counterfactual explanations
├── models/                 # DGN and RL agent implementations
├── utils/                  # Data loading, fingerprints, similarity, queues
├── Tox21Env.py             # RL environment for TOX21
├── data/                   # Dataset files (TOX21)
└── runs/                   # Outputs (TensorBoard logs, counterfactuals)

## Environment Setup

### Requirements

- Linux or macOS
- Miniconda or Anaconda
- CPU-based setup (recommended for development)

---

### Install the Environment

From the **root of the repository**, run:

```bash
bash install.sh

## This script will:

- create a conda environment named meg
- install Python 3.10
-install PyTorch 2.1 (CPU version)
- install PyTorch Geometric and all required dependencies
- install RDKit, TensorBoard, Typer, and scientific Python libraries

## Activate the environment: conda activate meg

## Training the DGN: 
python train_dgn.py tox21 <experiment_name>

This command trains a graph neural network on the TOX21 dataset and stores the trained model and logs under: runs/tox21/<experiment_name>/

## Generating Counterfactual Explanations MEG: python train_meg.py <experiment_name> --sample <ID>

## Rollout Video (MEG / Tox21)

This repo can optionally record a **rollout video** of the RL agent while it generates counterfactual molecules.
A video is a sequence of frames where **each frame corresponds to one environment step**.

### What the video shows
For each step, the video frame contains:
- the **current molecule** (the environment state **after** applying the chosen action)
- an overlay with:
  - `episode`, `step`, `steps_left`
  - `action`: the SMILES of the new molecule/state
  - `reward`: total reward (typically a combination of prediction + similarity)
  - `pred`: prediction-related reward component
  - `sim`: similarity-related reward component
  - `eps`: epsilon (exploration rate)

**Frame selection**
- When video recording is enabled, the code captures **1 frame per step** during selected episodes.
- With `--max-steps-per-episode 15`, the rollout video will contain up to **15 frames**.
- The FPS is controlled by `--video-fps` (e.g. 2 fps → ~7.5s for 15 steps).

### Output locations
For an experiment `tox22` (dataset `tox21`), outputs are saved under:

- TensorBoard logs:
  - `runs/tox21/tox22/plots/`
- Checkpoints / run artifacts:
  - `runs/tox21/tox22/ckpt/`
- Data splits:
  - `runs/tox21/tox22/splits/`
- Rollout videos (if enabled):
  - `runs/tox21/tox22/videos/`
  - example: `runs/tox21/tox22/videos/sample_1_ep_1.mp4`

### Run training (Tox21)
Basic run (no video):
```bash
python train_meg.py tox22 \
  --epochs 5000 \
  --max-steps-per-episode 15

### Run with rollout video recording:
python train_meg.py tox22 \
  --epochs 5000 \
  --max-steps-per-episode 15 \
  --record-video \
  --video-every 1 \
  --video-fps 2 \
  --video-size 448


### TensorBord
tensorboard --logdir runs/tox21/tox22/plots --port 6006 --bind_all
