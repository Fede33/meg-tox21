This repository contains a simplified and task-focused implementation of:

MEG: Generating Molecular Counterfactual Explanations for Deep Graph Networks
Numeroso & Bacciu, IJCNN 2021
https://arxiv.org/abs/2104.08060
 


This version is adapted to work exclusively on TOX21 (AhR task) and implements:

A Graph Neural Network (GCNN) for toxicity prediction

A Reinforcement Learning agent (Double DQN) for counterfactual search

A multi-objective reward:

prediction contrast (toxicity flip)

structural similarity (combined neural + Morgan fingerprint similarity)



Train the Toxicity GNN (DGN)
Train the GCNN model on TOX21:

python train_dgn.py tox21 flaminia


This will:
Create directory:

runs/tox21/exp/

Save:
Model checkpoint → ckpt/GCNN.pth
Dataset splits → splits/train.pth, val.pth, test.pth
TensorBoard logs → plots/
Hyperparameters → hyperparams.json


Generate Counterfactual Explanations (MEG)
Basic run (single sample)
python train_meg.py tox21 exp --sample 1

This generates counterfactuals for test molecule ID 1.

Run with more counterfactuals
python train_meg.py tox21 exp \
  --sample 1 \
  --num-counterfactuals 5

Example full run (as used in experiments)
python train_meg.py tox21 exp \
  --sample 90 \
  --epochs 100 \
  --num-counterfactuals 5 \
  --record-video \
  --record-n-episodes 5


Output Structure
After running MEG, results are saved in:
runs/tox21/exp/meg_output/<sample_id>/

Inside each sample folder:
data.json → original molecule + counterfactuals
embeddings/ → saved latent encodings
seed → random seed used
If video is enabled:
runs/tox21/exp/videos/



Evaluate Counterfactual Quality
To compute global metrics across generated counterfactuals:
python meg_metrics.py \
  --base runs/tox21/exp/meg_output \
  --k 10 \
  --save-cs

Metrics include:
Flip rate
Tox → Non-tox rate
Non-tox → Tox rate
Similarity statistics
Success@K


PLUS
TensorBoard (Training Monitoring)
For DGN training:
tensorboard --logdir runs/tox21/flaminia/plots --port 6006 --bind_all

For MEG runs:
tensorboard --logdir runs/tox21/flaminia/plots --port 6006 --bind_all

Tracked values:
RL reward
Prediction component
Similarity component
Training/validation accuracy (DGN)