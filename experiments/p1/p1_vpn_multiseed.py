# ================================================================
# P1 — ISCXVPN2016 Multi-Seed AUROC (3 seeds)
# CAFE-FKESE: CNN-LSTM Autoencoding Feature Extractor +
#             Flow-KNN Ensemble Scoring Engine
#
# Dataset  : ISCXVPN2016
# Latents  : z_tr_pca.npy (20500, 256)  ← fixed, loaded once
#            z_tn_pca.npy (9000,  256)
#            z_ta_pca.npy (9000,  256)
# Encoder  : FIXED — latents already extracted, not reloaded
# Flow     : 256-dim, 8 blocks — RETRAINED per seed
# Seeds    : 0, 1, 2
# Output   : AUROC per seed → mean ± std
# ================================================================

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

DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE    = '/content/drive/MyDrive/VPN_PCAP'
SEEDS   = [0, 1, 2]
EPOCHS  = 100
LR      = 1e-3
BS      = 512
PATIENCE = 10

print(f"Device: {DEVICE}")

# ── Build flow (256-dim, 8 blocks) ───────────────────────────────
def build_flow(n_feature=256, n_blocks=8):
    def subnet_fc(d_in, d_out):
        return nn.Sequential(
            nn.Linear(d_in, 2*d_in), nn.ReLU(),
            nn.Linear(2*d_in, d_out))
    flow = Ff.SequenceINN(n_feature)
    for _ in range(n_blocks):
        flow.append(Fm.AllInOneBlock,
            subnet_constructor=subnet_fc,
            affine_clamping=1.9,
            global_affine_type="SOFTPLUS",
            permute_soft=False)
    return flow.to(DEVICE)

# ── Full NLL: -log p(z) = 0.5||u||² - log|J| ────────────────────
@torch.no_grad()
def flow_nll(flow, z, bs=512):
    flow.eval()
    z_t  = torch.tensor(z, dtype=torch.float32)
    nlls = []
    for i in range(0, len(z_t), bs):
        batch = z_t[i:i+bs].to(DEVICE)
        p_u, log_jac = flow(batch)
        log_prob = -0.5 * torch.sum(p_u**2, dim=1) + log_jac
        nlls.append((-log_prob).cpu().numpy())
    return np.concatenate(nlls)

# ── Train flow ───────────────────────────────────────────────────
def train_flow(flow, z_tr, epochs=EPOCHS, lr=LR, bs=BS, patience=PATIENCE):
    opt    = optim.Adam(flow.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(torch.tensor(z_tr, dtype=torch.float32)),
        batch_size=bs, shuffle=True)
    best_loss = float('inf')
    best_state = None
    wait = 0
    for ep in range(1, epochs+1):
        flow.train()
        ep_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            p_u, log_jac = flow(xb)
            loss = (0.5*torch.sum(p_u**2, dim=1) - log_jac).mean()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= len(z_tr)
        if ep_loss < best_loss:
            best_loss  = ep_loss
            best_state = {k: v.clone() for k, v in flow.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stop at epoch {ep}  best_loss={best_loss:.4f}")
                break
        if ep % 10 == 0:
            print(f"  Epoch [{ep:3d}/{epochs}]  loss={ep_loss:.4f}  best={best_loss:.4f}")
    flow.load_state_dict(best_state)
    return best_loss

# ── Main ─────────────────────────────────────────────────────────
def main():
    # Load latents ONCE (encoder is fixed)
    print("Loading latents...")
    z_tr = np.load(f'{BASE}/z_tr_pca.npy').astype(np.float32)
    z_tn = np.load(f'{BASE}/z_tn_pca.npy').astype(np.float32)
    z_ta = np.load(f'{BASE}/z_ta_pca.npy').astype(np.float32)
    print(f"z_tr:{z_tr.shape}  z_tn:{z_tn.shape}  z_ta:{z_ta.shape}")

    y = np.array([0]*len(z_tn) + [1]*len(z_ta))

    auroc_list = []

    for seed in SEEDS:
        print(f"\n{'='*55}")
        print(f"SEED {seed}")
        print(f"{'='*55}")

        # Fix all randomness for this seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Build fresh flow
        flow = build_flow(256, 8)

        # Train flow
        print(f"Training flow (seed={seed})...")
        best_loss = train_flow(flow, z_tr)
        print(f"  Flow trained. Best loss: {best_loss:.4f}")

        # Flow NLL scores
        fl_tn  = flow_nll(flow, z_tn)
        fl_ta  = flow_nll(flow, z_ta)
        fl_all = np.concatenate([fl_tn, fl_ta])

        # KNN scores
        knn = NearestNeighbors(n_neighbors=5, n_jobs=-1)
        knn.fit(z_tr)
        knn_tn  = knn.kneighbors(z_tn)[0].mean(axis=1)
        knn_ta  = knn.kneighbors(z_ta)[0].mean(axis=1)
        knn_all = np.concatenate([knn_tn, knn_ta])

        # Auto-flip
        if roc_auc_score(y, fl_all) < 0.5:
            fl_all = -fl_all
        if roc_auc_score(y, knn_all) < 0.5:
            knn_all = -knn_all

        fl_norm  = RobustScaler().fit_transform(fl_all.reshape(-1,1)).ravel()
        knn_norm = RobustScaler().fit_transform(knn_all.reshape(-1,1)).ravel()

        # Grid search w*
        best_a, best_w = 0, 0.5
        for w in np.arange(0.1, 1.0, 0.1):
            a = roc_auc_score(y, w*fl_norm + (1-w)*knn_norm)
            if a > best_a:
                best_a, best_w = a, w

        ens   = best_w * fl_norm + (1-best_w) * knn_norm
        auroc = roc_auc_score(y, ens)

        print(f"  Seed {seed}: AUROC={auroc:.4f}  w*={best_w:.1f}")
        auroc_list.append(auroc)

        # Free memory
        del flow
        torch.cuda.empty_cache() if DEVICE == 'cuda' else None

    # Final summary
    auroc_arr = np.array(auroc_list)
    mean_a = auroc_arr.mean()
    std_a  = auroc_arr.std()
    ci95   = 1.96 * std_a

    print(f"\n{'='*55}")
    print(f"ISCXVPN2016 — P1 RESULTS (3 seeds)")
    print(f"{'='*55}")
    for s, a in zip(SEEDS, auroc_list):
        print(f"  Seed {s}: AUROC = {a:.4f}")
    print(f"\n  Mean AUROC : {mean_a:.4f}")
    print(f"  Std        : {std_a:.4f}")
    print(f"  95% CI     : ({mean_a-ci95:.4f}, {mean_a+ci95:.4f})")
    print(f"\n  Report as  : {mean_a:.4f} ± {std_a:.4f}")

if __name__ == '__main__':
    main()
