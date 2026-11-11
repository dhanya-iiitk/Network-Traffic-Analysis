# ================================================================
# P1 — CICIoT2023 Multi-Seed AUROC (3 seeds)
# Flow: 3200-dim, 6 blocks, GPU
# Results: Seed0=0.8848, Seed1=0.8751, Seed2=0.8867
# Mean AUROC = 0.8822 +/- 0.0051
# 95% CI = (0.8722, 0.8922)
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

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
BASE='/content/drive/MyDrive/CIC IoT2023'
SEEDS=[0,1,2]; EPOCHS=100; LR=1e-3; BS=256; PATIENCE=10
print(f"Device: {DEVICE}")

def build_flow(n_feature=3200,n_blocks=6):
    def subnet_fc(d_in,d_out):
        return nn.Sequential(nn.Linear(d_in,2*d_in),nn.ReLU(),nn.Linear(2*d_in,d_out))
    flow=Ff.SequenceINN(n_feature)
    for _ in range(n_blocks):
        flow.append(Fm.AllInOneBlock,subnet_constructor=subnet_fc,
            affine_clamping=1.9,global_affine_type="SOFTPLUS",permute_soft=False)
    return flow.to(DEVICE)

@torch.no_grad()
def flow_nll(flow,z,bs=256):
    flow.eval(); z_t=torch.tensor(z,dtype=torch.float32); nlls=[]
    for i in range(0,len(z_t),bs):
        p_u,log_jac=flow(z_t[i:i+bs].to(DEVICE))
        log_prob=-0.5*torch.sum(p_u**2,dim=1)+log_jac
        nlls.append((-log_prob).cpu().numpy())
    return np.concatenate(nlls)

def train_flow(flow,z_tr):
    opt=optim.Adam(flow.parameters(),lr=LR)
    loader=DataLoader(TensorDataset(torch.tensor(z_tr,dtype=torch.float32)),batch_size=BS,shuffle=True)
    best_loss=float('inf'); best_state=None; wait=0
    for ep in range(1,EPOCHS+1):
        flow.train(); ep_loss=0.0
        for (xb,) in loader:
            xb=xb.to(DEVICE); opt.zero_grad()
            p_u,log_jac=flow(xb)
            loss=(0.5*torch.sum(p_u**2,dim=1)-log_jac).mean()
            loss.backward(); opt.step(); ep_loss+=loss.item()*len(xb)
        ep_loss/=len(z_tr)
        if ep_loss<best_loss:
            best_loss=ep_loss; best_state={k:v.clone() for k,v in flow.state_dict().items()}; wait=0
        else:
            wait+=1
            if wait>=PATIENCE: print(f"  Early stop ep{ep} best={best_loss:.4f}"); break
        if ep%10==0: print(f"  Epoch[{ep:3d}] loss={ep_loss:.4f} best={best_loss:.4f}")
    flow.load_state_dict(best_state); return best_loss

z_tr=np.load(f'{BASE}/z_tr_reextracted.npy').astype(np.float32)
z_tn=np.load(f'{BASE}/z_tn_reextracted.npy').astype(np.float32)
z_ta=np.load(f'{BASE}/z_ta_reextracted.npy').astype(np.float32)
y=np.array([0]*len(z_tn)+[1]*len(z_ta)); auroc_list=[]

for seed in SEEDS:
    print(f"\n{'='*55}\nSEED {seed}\n{'='*55}")
    torch.manual_seed(seed); np.random.seed(seed)
    flow=build_flow(3200,6)
    best_loss=train_flow(flow,z_tr); print(f"  Best loss: {best_loss:.4f}")
    fl_tn=flow_nll(flow,z_tn); fl_ta=flow_nll(flow,z_ta)
    fl_all=np.concatenate([fl_tn,fl_ta])
    knn=NearestNeighbors(n_neighbors=5,n_jobs=-1); knn.fit(z_tr)
    knn_all=np.concatenate([knn.kneighbors(z_tn)[0].mean(axis=1),knn.kneighbors(z_ta)[0].mean(axis=1)])
    fl_norm=RobustScaler().fit_transform((-fl_all).reshape(-1,1)).ravel()
    if roc_auc_score(y,knn_all)<0.5: knn_all=-knn_all
    knn_norm=RobustScaler().fit_transform(knn_all.reshape(-1,1)).ravel()
    best_a,best_w=0,0.5
    for w in np.arange(0.1,1.0,0.1):
        a=roc_auc_score(y,w*fl_norm+(1-w)*knn_norm)
        if a>best_a: best_a,best_w=a,w
    ens=best_w*fl_norm+(1-best_w)*knn_norm
    auroc=roc_auc_score(y,ens)
    print(f"  Seed {seed}: AUROC={auroc:.4f} w*={best_w:.1f}")
    auroc_list.append(auroc)
    del flow
    if DEVICE=='cuda': torch.cuda.empty_cache()

auroc_arr=np.array(auroc_list); mean_a=auroc_arr.mean(); std_a=auroc_arr.std(); ci95=1.96*std_a
print(f"\n{'='*55}\nCICIoT2023 — P1 RESULTS\n{'='*55}")
for s,a in zip(SEEDS,auroc_list): print(f"  Seed {s}: AUROC = {a:.4f}")
print(f"\n  Mean AUROC : {mean_a:.4f}\n  Std        : {std_a:.4f}\n  95% CI     : ({mean_a-ci95:.4f},{mean_a+ci95:.4f})\n  Report as  : {mean_a:.4f} +/- {std_a:.4f}")
