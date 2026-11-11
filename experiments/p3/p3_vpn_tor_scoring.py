# ================================================================
# P3 — ISCXVPN2016 & ISCXTor2016 Low-FPR Scoring
# VPN Result: AUROC=0.9234, pAUROC=0.3763
# TOR Result: AUROC=0.8937, pAUROC=0.6528
# Both use 256-dim PCA latents, 8 flow blocks
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

def build_flow(n_feature, n_blocks=8):
    def subnet_fc(d_in, d_out):
        return nn.Sequential(nn.Linear(d_in, 2*d_in), nn.ReLU(), nn.Linear(2*d_in, d_out))
    flow = Ff.SequenceINN(n_feature)
    for _ in range(n_blocks):
        flow.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc,
            affine_clamping=1.9, global_affine_type="SOFTPLUS", permute_soft=False)
    return flow

@torch.no_grad()
def flow_nll(flow, z, bs=512):
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

def score_dataset(base_path, name):
    print(f"\n{'='*50}\nScoring: {name}\n{'='*50}")
    z_tr = np.load(f'{base_path}/z_tr_pca.npy').astype(np.float32)
    z_tn = np.load(f'{base_path}/z_tn_pca.npy').astype(np.float32)
    z_ta = np.load(f'{base_path}/z_ta_pca.npy').astype(np.float32)
    print(f"z_tr:{z_tr.shape}, z_tn:{z_tn.shape}, z_ta:{z_ta.shape}")

    flow = build_flow(256, n_blocks=8)
    flow.load_state_dict(torch.load(f'{base_path}/best_flow.pth', map_location='cpu'))
    flow.eval()

    fl_all  = np.concatenate([flow_nll(flow,z_tn), flow_nll(flow,z_ta)])
    knn     = NearestNeighbors(n_neighbors=5, n_jobs=-1).fit(z_tr)
    knn_all = np.concatenate([knn.kneighbors(z_tn)[0].mean(1), knn.kneighbors(z_ta)[0].mean(1)])
    y       = np.array([0]*len(z_tn) + [1]*len(z_ta))

    if roc_auc_score(y, fl_all)  < 0.5: fl_all  = -fl_all
    if roc_auc_score(y, knn_all) < 0.5: knn_all = -knn_all

    fl_norm  = RobustScaler().fit_transform(fl_all.reshape(-1,1)).ravel()
    knn_norm = RobustScaler().fit_transform(knn_all.reshape(-1,1)).ravel()

    best_a, best_w = 0, 0.5
    for w in np.arange(0.1, 1.0, 0.1):
        a = roc_auc_score(y, w*fl_norm + (1-w)*knn_norm)
        if a > best_a: best_a, best_w = a, w

    ens   = best_w*fl_norm + (1-best_w)*knn_norm
    auroc = roc_auc_score(y, ens)
    fpr, tpr, _ = roc_curve(y, ens)
    mask = fpr <= 0.05
    pauc = trapezoid(tpr[mask], fpr[mask]) / 0.05

    print(f"AUROC={auroc:.4f}  pAUROC={pauc:.4f}  w*={best_w:.1f}")
    print(f"TPR@0.1%={tpr_at_fpr(fpr,tpr,0.001):.4f}  TPR@0.5%={tpr_at_fpr(fpr,tpr,0.005):.4f}")
    print(f"TPR@1%={tpr_at_fpr(fpr,tpr,0.01):.4f}  TPR@5%={tpr_at_fpr(fpr,tpr,0.05):.4f}")

score_dataset('/content/drive/MyDrive/VPN_PCAP', 'ISCXVPN2016')
score_dataset('/content/drive/MyDrive/TOR_PCAP', 'ISCXTor2016')
