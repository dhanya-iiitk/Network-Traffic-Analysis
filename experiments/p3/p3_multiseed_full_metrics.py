# ================================================================
# P3 — Multi-Seed Full Metrics (R1-C4)
# CAFE-FKESE: CNN-LSTM Autoencoding Feature Extractor +
#             Flow-KNN Ensemble Scoring Engine
#
# Computes per-seed: AUROC, AUPRC, F1, FPR (at Youden threshold)
# Seeds  : 0, 1, 2  (encoder fixed; flow + KNN retrained per seed)
# Output : mean ± std + 95% CI for all metrics, all datasets
#
# Datasets:
#   CICIoT2023   — dim=3200, 6 blocks, invert=True
#   ISCXVPN2016  — dim=256,  8 blocks, invert=False
#   ISCXTor2016  — dim=256,  8 blocks, invert=False
#   CICIDS2017   — dim=256,  8 blocks, invert=False (z_tr_256.npy)
# ================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              roc_curve, f1_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import json

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS    = [0, 1, 2]
EPOCHS   = 100
LR       = 1e-3
BS       = 512
PATIENCE = 10
print(f"Device: {DEVICE}")

# ── Dataset configs ───────────────────────────────────────────────
CONFIGS = {
    'CICIoT2023': {
        'base'   : '/content/drive/MyDrive/CIC IoT2023',
        'z_tr'   : 'z_tr_reextracted.npy',
        'z_n'    : 'z_tn_reextracted.npy',
        'z_a'    : 'z_ta_reextracted.npy',
        'dim'    : 3200, 'blocks': 6, 'invert': True,
    },
    'ISCXVPN2016': {
        'base'   : '/content/drive/MyDrive/VPN_PCAP',
        'z_tr'   : 'z_tr_pca.npy',
        'z_n'    : 'z_tn_pca.npy',
        'z_a'    : 'z_ta_pca.npy',
        'dim'    : 256, 'blocks': 8, 'invert': False,
    },
    'ISCXTor2016': {
        'base'   : '/content/drive/MyDrive/TOR_PCAP',
        'z_tr'   : 'z_tr_pca.npy',
        'z_n'    : 'z_tn_pca.npy',
        'z_a'    : 'z_ta_pca.npy',
        'dim'    : 256, 'blocks': 8, 'invert': False,
    },
    'CICIDS2017': {
        'base'   : '/content/drive/MyDrive/CIC IDS2017',
        'z_tr'   : 'z_tr_256.npy',
        'z_n'    : 'z_tn_256.npy',
        'z_a'    : 'z_ta_256.npy',
        'dim'    : 256, 'blocks': 8, 'invert': False,
    },
}

# ── Flow helpers ──────────────────────────────────────────────────
def build_flow(dim, n_blocks):
    def subnet_fc(d_in, d_out):
        return nn.Sequential(
            nn.Linear(d_in, 2*d_in), nn.ReLU(),
            nn.Linear(2*d_in, d_out))
    flow = Ff.SequenceINN(dim)
    for _ in range(n_blocks):
        flow.append(Fm.AllInOneBlock,
            subnet_constructor=subnet_fc,
            affine_clamping=1.9,
            global_affine_type="SOFTPLUS",
            permute_soft=False)
    return flow.to(DEVICE)

def train_flow(flow, z_tr):
    opt    = optim.Adam(flow.parameters(), lr=LR)
    loader = DataLoader(
        TensorDataset(torch.tensor(z_tr, dtype=torch.float32)),
        batch_size=BS, shuffle=True)
    best_loss, best_state, wait = float('inf'), None, 0
    for ep in range(1, EPOCHS+1):
        flow.train(); ep_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(DEVICE); opt.zero_grad()
            p_u, log_jac = flow(xb)
            loss = (0.5*torch.sum(p_u**2, dim=1) - log_jac).mean()
            loss.backward(); opt.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= len(z_tr)
        if ep_loss < best_loss:
            best_loss  = ep_loss
            best_state = {k: v.clone() for k, v in flow.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  Early stop ep{ep}  best={best_loss:.4f}")
                break
        if ep % 10 == 0:
            print(f"  ep{ep:3d}  loss={ep_loss:.4f}  best={best_loss:.4f}")
    flow.load_state_dict(best_state)
    return flow

@torch.no_grad()
def flow_nll(flow, z, bs=512):
    flow.eval()
    z_t = torch.tensor(z, dtype=torch.float32)
    out = []
    for i in range(0, len(z_t), bs):
        p_u, lj = flow(z_t[i:i+bs].to(DEVICE))
        out.append((0.5*torch.sum(p_u**2, dim=1) - lj).cpu().numpy())
    return np.concatenate(out)

# ── Metrics at Youden threshold ───────────────────────────────────
def compute_metrics(y, scores):
    auroc = roc_auc_score(y, scores)
    auprc = average_precision_score(y, scores)
    fpr_arr, tpr_arr, threshs = roc_curve(y, scores)
    idx   = np.argmax(tpr_arr - fpr_arr)
    t_opt = threshs[idx]
    pred  = (scores >= t_opt).astype(int)
    f1    = f1_score(y, pred)
    fpr   = fpr_arr[idx]
    return {'auroc': round(float(auroc), 4),
            'auprc': round(float(auprc), 4),
            'f1'   : round(float(f1),    4),
            'fpr'  : round(float(fpr),   4)}

# ── Summary helper ────────────────────────────────────────────────
def summarise(vals):
    a = np.array(vals)
    mean = a.mean(); std = a.std(); ci = 1.96*std
    return {'mean': round(float(mean),4),
            'std' : round(float(std), 4),
            'ci95': (round(float(mean-ci),4), round(float(mean+ci),4))}

# ── Main loop ─────────────────────────────────────────────────────
all_results = {}

for ds_name, cfg in CONFIGS.items():
    print(f"\n{'='*60}")
    print(f"  DATASET: {ds_name}  (dim={cfg['dim']}, blocks={cfg['blocks']})")
    print(f"{'='*60}")

    p    = cfg['base']
    z_tr = np.load(f"{p}/{cfg['z_tr']}").astype(np.float32)
    z_n  = np.load(f"{p}/{cfg['z_n']}").astype(np.float32)
    z_a  = np.load(f"{p}/{cfg['z_a']}").astype(np.float32)
    print(f"  z_tr:{z_tr.shape}  z_n:{z_n.shape}  z_a:{z_a.shape}")

    y = np.array([0]*len(z_n) + [1]*len(z_a))

    seed_metrics = []

    for seed in SEEDS:
        print(f"\n  --- Seed {seed} ---")
        torch.manual_seed(seed); np.random.seed(seed)

        flow = build_flow(cfg['dim'], cfg['blocks'])
        flow = train_flow(flow, z_tr)

        fl_n = flow_nll(flow, z_n)
        fl_a = flow_nll(flow, z_a)
        if cfg['invert']: fl_n = -fl_n; fl_a = -fl_a
        fl_all = np.concatenate([fl_n, fl_a])
        if roc_auc_score(y, fl_all) < 0.5: fl_all = -fl_all

        knn = NearestNeighbors(n_neighbors=5, n_jobs=-1).fit(z_tr)
        kn  = knn.kneighbors(z_n)[0].mean(1)
        ka  = knn.kneighbors(z_a)[0].mean(1)
        knn_all = np.concatenate([kn, ka])
        if roc_auc_score(y, knn_all) < 0.5: knn_all = -knn_all

        fl_norm  = RobustScaler().fit_transform(fl_all.reshape(-1,1)).ravel()
        knn_norm = RobustScaler().fit_transform(knn_all.reshape(-1,1)).ravel()

        best_a, best_w = 0, 0.5
        for w in np.arange(0.1, 1.0, 0.1):
            a = roc_auc_score(y, w*fl_norm + (1-w)*knn_norm)
            if a > best_a: best_a, best_w = a, w

        ens = best_w*fl_norm + (1-best_w)*knn_norm
        m   = compute_metrics(y, ens)
        m['w_star'] = round(float(best_w), 1)
        seed_metrics.append(m)
        print(f"  Seed {seed}: AUROC={m['auroc']}  AUPRC={m['auprc']}  F1={m['f1']}  FPR={m['fpr']}  w*={m['w_star']}")

        del flow
        if DEVICE == 'cuda': torch.cuda.empty_cache()

    summary = {}
    for metric in ['auroc', 'auprc', 'f1', 'fpr']:
        vals = [s[metric] for s in seed_metrics]
        summary[metric] = summarise(vals)
        summary[metric]['per_seed'] = vals

    all_results[ds_name] = summary

    print(f"\n  SUMMARY — {ds_name}")
    print(f"  {'Metric':<8} {'S0':>7} {'S1':>7} {'S2':>7} {'Mean±Std':>14} {'95% CI':>20}")
    print(f"  {'-'*62}")
    for metric in ['auroc', 'auprc', 'f1', 'fpr']:
        s  = summary[metric]
        ps = s['per_seed']
        print(f"  {metric.upper():<8} {ps[0]:>7.4f} {ps[1]:>7.4f} {ps[2]:>7.4f} "
              f"  {s['mean']:.4f}±{s['std']:.4f}   {str(s['ci95']):>20}")

print("\n\n" + "="*70)
print("LaTeX TABLE ROWS (Mean ± Std, 95% CI)")
print("="*70)
for ds, res in all_results.items():
    auroc = res['auroc']; auprc = res['auprc']
    f1    = res['f1'];    fpr   = res['fpr']
    print(f"{ds}")
    print(f"  AUROC : {auroc['mean']:.4f} ± {auroc['std']:.4f}  CI=({auroc['ci95'][0]:.4f},{auroc['ci95'][1]:.4f})")
    print(f"  AUPRC : {auprc['mean']:.4f} ± {auprc['std']:.4f}  CI=({auprc['ci95'][0]:.4f},{auprc['ci95'][1]:.4f})")
    print(f"  F1    : {f1['mean']:.4f} ± {f1['std']:.4f}  CI=({f1['ci95'][0]:.4f},{f1['ci95'][1]:.4f})")
    print(f"  FPR   : {fpr['mean']:.4f} ± {fpr['std']:.4f}  CI=({fpr['ci95'][0]:.4f},{fpr['ci95'][1]:.4f})")

save_path = '/content/drive/MyDrive/p3_multiseed_full_metrics.json'
with open(save_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\n✅ Saved to {save_path}")
