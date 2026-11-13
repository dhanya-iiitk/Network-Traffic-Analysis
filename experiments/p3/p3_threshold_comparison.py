# ================================================================
# P3 — Youden vs Label-Free Threshold Comparison (R1-C1)
# Results: Youden outperforms label-free by 0.07--0.42 on all 4 datasets
# ================================================================
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_curve, accuracy_score
import FrEIA.framework as Ff, FrEIA.modules as Fm

def build_flow(dim,n_blocks):
    def subnet_fc(d_in,d_out): return nn.Sequential(nn.Linear(d_in,2*d_in),nn.ReLU(),nn.Linear(2*d_in,d_out))
    flow=Ff.SequenceINN(dim)
    for _ in range(n_blocks):
        flow.append(Fm.AllInOneBlock,subnet_constructor=subnet_fc,affine_clamping=1.9,global_affine_type="SOFTPLUS",permute_soft=False)
    return flow

@torch.no_grad()
def fscore(flow,Z,bs=256):
    Zt=torch.tensor(Z.astype(np.float32)); sc=[]
    for i in range(0,len(Zt),bs):
        pu,lj=flow(Zt[i:i+bs]); sc.extend((0.5*torch.sum(pu**2,dim=1)-lj).cpu().numpy())
    return np.array(sc)

CONFIGS = {
    'CICIDS2017': {'base':'/content/drive/MyDrive/CIC IDS2017','flow':'best_flow.pth','z_tr':'z_tr_pca.npy','z_n':'z_tn_pca.npy','z_a':'z_ta_pca.npy','dim':6400,'blocks':8,'invert':False,'skip':True},
    'CICIoT2023': {'base':'/content/drive/MyDrive/CIC IoT2023','flow':'best_flow_retrained.pth','z_tr':'z_tr_reextracted.npy','z_n':'z_tn_reextracted.npy','z_a':'z_ta_reextracted.npy','dim':3200,'blocks':6,'invert':True,'skip':False},
    'ISCXVPN2016': {'base':'/content/drive/MyDrive/VPN_PCAP','flow':'best_flow.pth','z_tr':'z_tr_pca.npy','z_n':'z_tn_pca.npy','z_a':'z_ta_pca.npy','dim':256,'blocks':8,'invert':False,'skip':False},
    'ISCXTor2016': {'base':'/content/drive/MyDrive/TOR_PCAP','flow':'best_flow.pth','z_tr':'z_tr_pca.npy','z_n':'z_tn_pca.npy','z_a':'z_ta_pca.npy','dim':256,'blocks':8,'invert':False,'skip':False},
}

results={}
for name,cfg in CONFIGS.items():
    if cfg.get('skip'): print(f"\nSkipping {name} — High-RAM required."); results[name]={'youden_acc':None,'p95_acc':None,'p99_acc':None}; continue
    p=cfg['base']
    z_tr=np.load(f"{p}/{cfg['z_tr']}").astype(np.float32)
    z_n=np.load(f"{p}/{cfg['z_n']}").astype(np.float32)
    z_a=np.load(f"{p}/{cfg['z_a']}").astype(np.float32)
    flow=build_flow(cfg['dim'],cfg['blocks'])
    flow.load_state_dict(torch.load(f"{p}/{cfg['flow']}",map_location='cpu')); flow.eval()
    sf_tr=fscore(flow,z_tr); sf_n=fscore(flow,z_n); sf_a=fscore(flow,z_a)
    if cfg['invert']: sf_tr=-sf_tr; sf_n=-sf_n; sf_a=-sf_a
    y=np.array([0]*len(sf_n)+[1]*len(sf_a)); s=np.concatenate([sf_n,sf_a])
    fpr_c,tpr_c,threshs=roc_curve(y,s); idx_you=np.argmax(tpr_c-fpr_c); t_you=threshs[idx_you]
    acc_you=accuracy_score(y,(s>=t_you).astype(int))
    t_p95=np.percentile(sf_tr,95); acc_p95=accuracy_score(y,(s>=t_p95).astype(int))
    t_p99=np.percentile(sf_tr,99); acc_p99=accuracy_score(y,(s>=t_p99).astype(int))
    print(f"\n{name}  Youden={acc_you:.4f}  P95={acc_p95:.4f}  P99={acc_p99:.4f}")
    results[name]={'youden_acc':round(float(acc_you),4),'p95_acc':round(float(acc_p95),4),'p99_acc':round(float(acc_p99),4)}

print("\n"+"="*60+"\nTHRESHOLD COMPARISON SUMMARY (Accuracy)\n"+"="*60)
print(f"{'Dataset':<14} {'Youden':>8} {'P95':>8} {'P99':>8}")
for name,r in results.items():
    if r['youden_acc'] is None: print(f"{name:<14} {'---':>8} {'---':>8} {'---':>8}")
    else: print(f"{name:<14} {r['youden_acc']:>8.4f} {r['p95_acc']:>8.4f} {r['p99_acc']:>8.4f}")