# ================================================================
# P1 — CICIDS2017 Multi-Seed AUROC
# Flow: 6400-dim, 8 blocks, CPU only
# NOTE: Multi-seed not feasible — model + Adam optimizer
#       requires ~13 GB RAM, exceeding Colab limit.
#       Single-run AUROC = 0.9655 reported (from P3).
# ================================================================
# This script is provided for reference only.
# Run on a machine with 32+ GB RAM for multi-seed results.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
import FrEIA.framework as Ff
import FrEIA.modules as Fm

DEVICE='cpu'
BASE='/content/drive/MyDrive/CIC IDS2017'
SEEDS=[0,1,2]; EPOCHS=100; LR=1e-3; BS=64; PATIENCE=10
print(f"Device: {DEVICE}")
print("WARNING: This script requires 32+ GB RAM. May crash on Colab.")
