# MEG for Toxicity Counterfactual Explanations on TOX21

This repository contains a simplified and task-focused implementation of **MEG: Generating Molecular Counterfactual Explanations for Deep Graph Networks** adapted to the **TOX21 AhR toxicity task**.

The project was developed for the **Reinforcement Learning course** and combines **graph neural networks** with **reinforcement learning** to generate molecular counterfactual explanations.

More specifically, the pipeline first trains a graph-based toxicity predictor and then uses a reinforcement learning agent to search for structurally similar molecules whose predicted toxicity class is flipped.

---

## Project Overview

The project is built around two main components:

### 1. Toxicity Prediction Model
A **Graph Convolutional Neural Network (GCNN)** is trained on the **TOX21 AhR** dataset to perform binary toxicity classification.

The model takes molecular graphs as input and predicts whether a molecule is toxic or non-toxic.

### 2. Counterfactual Generation with Reinforcement Learning
A **Double DQN** agent is used to search for molecular counterfactuals.

Starting from an original molecule, the agent explores valid molecular edits and tries to generate alternative molecules that:

- **flip the predicted class**
- remain **structurally similar** to the original molecule

This makes the explanation both **contrastive** and **chemically meaningful**.

---

## Method

The implementation follows a simplified MEG-style pipeline specialized for **TOX21**.

### Toxicity Classifier
The toxicity predictor is a **GCNN** built with **PyTorch Geometric**.

Its architecture includes:

- 3 graph convolution layers (`GraphConv`)
- global max pooling and global mean pooling
- fully connected layers for graph-level classification

The model outputs:

- the toxicity logits
- node embeddings
- graph embeddings

These learned embeddings are later reused for similarity computation in the counterfactual search phase.

---

## Reinforcement Learning for Counterfactual Search

Counterfactual generation is formulated as a sequential decision-making problem.

A **Double DQN** agent interacts with a molecular editing environment and selects actions corresponding to valid graph/molecule modifications.

The RL agent optimizes a **multi-objective reward** combining:

- **prediction contrast**: how much the edited molecule moves toward the opposite toxicity class
- **structural similarity**: how similar the candidate molecule remains to the original one

The reward used in the TOX21 setting combines:

\[
reward = (1 - w_{sim}) \cdot reward_{pred} + w_{sim} \cdot reward_{sim}
\]

where:

- `reward_pred` encourages class flipping
- `reward_sim` encourages similarity preservation
- `w_sim` controls the trade-off between the two objectives

---

## Similarity Measures

The project supports different similarity functions for molecular comparison:

- **Tanimoto similarity** on Morgan fingerprints
- **Cosine similarity** on neural graph embeddings
- **Rescaled neural similarity**
- **Combined similarity**

The **combined similarity** used in the main TOX21 setup is:

\[
0.5 \cdot \text{cosine(neural embedding)} + 0.5 \cdot \text{Tanimoto(Morgan fingerprint)}
\]

This allows the search process to balance:

- similarity in **learned representation space**
- similarity in **classical cheminformatics fingerprint space**

---

## Data Processing

The project works exclusively on **TOX21 (AhR task)**.

The preprocessing pipeline includes:

- loading `Tox21_AhR_training`, `Tox21_AhR_evaluation`, and `Tox21_AhR_testing`
- conversion to **PyTorch Geometric** graph format
- feature padding
- molecule validity filtering using **RDKit sanitization**
- class balancing by undersampling the negative class
- saving experiment-specific train/validation/test splits

Saved splits are stored under: `runs/tox21/<experiment>/splits/`


## Molecular Representation

Molecules are represented both as:

- **RDKit molecular objects**
- **PyTorch Geometric graphs**

The project includes utilities for:

- RDKit ↔ PyG conversion
- atom-type encoding with a **53-class one-hot mapping**
- bond-type encoding
- Morgan **bit fingerprints**
- Morgan **count fingerprints**
- RDKit fingerprints

This allows the system to move seamlessly between:

- graph-based learning
- symbolic molecular editing
- similarity computation

---

## Main Components

### Toxicity Prediction

- **`GCNN`** — graph neural network used as the toxicity predictor  
- **`train_dgn.py`** — training script for the toxicity classifier  
- **`train_cycle_classifier()`** — training/validation/test loop with checkpointing and metrics  
- **`plot_history.py`** — utility to visualize training loss and accuracy curves  

---

### Reinforcement Learning / Counterfactual Search

- **`train_meg.py`** — main script for counterfactual generation  
- **`CF_Tox21`** — molecular environment used for toxicity counterfactual search  
- **`Agent`** — Double DQN agent responsible for exploring molecular edits  
- **`MolDQN`** — Q-network used by the RL agent  
- **`ReplayMemory`** — replay buffer for experience replay  
- **`SortedQueue`** — stores the top-k generated counterfactual molecules ranked by reward  

---

### Similarity and Molecular Utilities

The project includes utilities for computing molecular similarity using:

- **Morgan fingerprints**
- **RDKit molecular conversions**
- **combined neural + fingerprint similarity**
- **TOX21 graph conversion utilities**

---

## Evaluation

The project includes evaluation tools to measure the quality of generated counterfactual explanations.

- **`meg_metrics.py`** — computes global counterfactual quality metrics  
- **`compute_cf_metrics_tox21()`** — computes per-sample counterfactual metrics  
- **TensorBoard logging** for both toxicity prediction and MEG runs  

---

## Evaluation Metrics

The counterfactual explanations are evaluated using several metrics, including:

- **Flip Rate**
- **Tox → Non-tox Rate**
- **Non-tox → Tox Rate**
- **Success@K**
- **Similarity statistics**
- **Flip rate at different similarity thresholds**
- **Prediction contrast statistics**

The evaluation script can also export:

- aggregate summaries
- pair-level tables
- Pareto-style rankings of generated counterfactual molecules



## Technologies Used

- Python
- PyTorch
- PyTorch Geometric
- RDKit
- NumPy
- pandas
- matplotlib
- TensorBoard
- Typer

---

## Learning Objectives

This project focuses on:

- using **Graph Neural Networks (GNNs)** for molecular property prediction
- applying **reinforcement learning** to structured search problems
- generating **counterfactual explanations** for graph-based models
- combining **prediction contrast** and **structural similarity** in a multi-objective reward
- evaluating explanation quality using both **local and global metrics**
- working with molecular data using **RDKit** and **PyTorch Geometric**

---
## How to run

## Generate Counterfactual Explanations

### Basic run on one sample

`python train_meg.py <experiment_name> --sample 1`

Example:

`python train_meg.py exp --sample 1`

### Run with multiple counterfactuals

`python train_meg.py exp --sample 1 --num-counterfactuals 5`

Full example run

`python train_meg.py exp --sample 90 --epochs 100 --num-counterfactuals 5 --record-video --record-n-episodes 5`

### Evaluate Generated Counterfactuals
`python meg_metrics.py --base runs/tox21/<experiment_name>/meg_output --k 10 --save-csv`

Example:

`python meg_metrics.py --base runs/tox21/exp/meg_output --k 10 --save-csv`

### Plot DGN Training History

`python plot_history.py --run_dir ./runs/tox21/<experiment_name>`

Example:

`python plot_history.py --run_dir ./runs/tox21/exp`

### Monitor Training with TensorBoard

#### For toxicity predictor training

`tensorboard --logdir runs/tox21/<experiment_name>/plots --port 6006 --bind_all`

#### For MEG runs

`tensorboard --logdir runs/tox21/<experiment_name>/plots --port 6006 --bind_all`

