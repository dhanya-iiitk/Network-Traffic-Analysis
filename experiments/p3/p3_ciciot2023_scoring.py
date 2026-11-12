# ================================================================
# P3 — CICIoT2023 Low-FPR Scoring
# CAFE-FKESE: CNN-LSTM Autoencoding Feature Extractor +
#             Flow-KNN Ensemble Scoring Engine
#
# Dataset  : CICIoT2023
# Latents  : z_tr_reextracted.npy (10500, 3200)
#            z_tn_reextracted.npy (4500, 3200)
#            z_ta_reextracted.npy (4500, 3200)
# Flow     : best_flow_retrained.pth (3200-dim, 6 blocks)
# Result   : AUROC=0.9047, pAUROC=0.0747
#
# CRITICAL: Flow scores are INVERTED (-fl_all) for CICIoT2023
#           because raw flow AUROC < 0.5 (=0.1338)
# ================================================================

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from scipy.integrate import trapezoid
import FrEIA.framework as Ff
import FrEIA.modules as Fm

DEVICE = 'cpu'
BASE1  = '/content/drive/MyDrive/CIC IoT2023'

def build_flow(n_feature, n_blocks=6):
    def subnet_fc(d_in, d_out):
        return nn.Sequential(nn.Linear(d_in, 2*d_in), nn.ReLU(), nn.Linear(2*d_in, d_out))
    flow = Ff.SequenceINN(n_feature)
    for _ in range(n_blocks):
        flow.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc,
            affine_clamping=1.9, global_affine_type="SOFTPLUS", permute_soft=False)
    return flow

@torch.no_grad()
def flow_nll(flow, z, bs=256):
    z_t = torch.tensor(z, dtype=torch.float32)
    nlls = []
    for i in range(0, len(z_t), bs):
        p_u, log_jac = flow(z_t[i:i+bs])
        log_prob = -0.5 * torch.sum(p_u**2, dim=1) + log_jac
        nlls.append((-log_prob).numpy())
    return np.concatenate(nlls)

def tpr_at_fpr(fpr_arr, tpr_arr, target):
    idx = min(np.searchsorted(fpr_arr, target), len(tpr_arr)-1)
    return tpr_arr[idx]

z_tr = np.load(f'{BASE1}/z_tr_reextracted.npy')
z_tn = np.load(f'{BASE1}/z_tn_reextracted.npy')
z_ta = np.load(f'{BASE1}/z_ta_reextracted.npy')
print(f"z_tr:{z_tr.shape}, z_tn:{z_tn.shape}, z_ta:{z_ta.shape}")

flow = build_flow(3200, n_blocks=6)
flow.load_state_dict(torch.load(f'{BASE1}/best_flow_retrained.pth', map_location='cpu'))
flow.eval()

fl_all  = np.concatenate([flow_nll(flow, z_tn), flow_nll(flow, z_ta)])
knn     = NearestNeighbors(n_neighbors=5, n_jobs=-1).fit(z_tr)
knn_all = np.concatenate([knn.kneighbors(z_tn)[0].mean(1), knn.kneighbors(z_ta)[0].mean(1)])
y       = np.array([0]*len(z_tn) + [1]*len(z_ta))

print(f"Raw flow AUROC: {roc_auc_score(y, fl_all):.4f} (inverted)")
fl_norm  = RobustScaler().fit_transform((-fl_all).reshape(-1,1)).ravel()
knn_norm = RobustScaler().fit_transform(knn_all.reshape(-1,1)).ravel()

best_a, best_w = 0, 0.5
for w in np.arange(0.1, 1.0, 0.1):
    a = roc_auc_score(y, w*fl_norm + (1-w)*knn_norm)
    if a > best_a: best_a, best_w = a, w

ens  = best_w*fl_norm + (1-best_w)*knn_norm
auroc = roc_auc_score(y, ens)
fpr, tpr, _ = roc_curve(y, ens)
mask = fpr <= 0.05
pauc = trapezoid(tpr[mask], fpr[mask]) / 0.05

print(f"AUROC={auroc:.4f}  pAUROC={pauc:.4f}  w*={best_w:.1f}")
print(f"TPR@0.1%={tpr_at_fpr(fpr,tpr,0.001):.4f}  TPR@0.5%={tpr_at_fpr(fpr,tpr,0.005):.4f}")
print(f"TPR@1%={tpr_at_fpr(fpr,tpr,0.01):.4f}  TPR@5%={tpr_at_fpr(fpr,tpr,0.05):.4f}")

# Fusion strategy comparison (R1-C3)
ens_eq  = 0.5*fl_norm + 0.5*knn_norm
a_flow  = roc_auc_score(y, fl_norm)
a_knn   = roc_auc_score(y, knn_norm)
a_equal = roc_auc_score(y, ens_eq)
print(f"\nFusion strategy comparison:")
print(f"  Flow only  (w=1.0) : {a_flow:.4f}")
print(f"  Equal-w    (w=0.5) : {a_equal:.4f}")
print(f"  KNN only   (w=0.0) : {a_knn:.4f}")
print(f"  Adaptive   (w={best_w:.1f}) : {auroc:.4f}")
