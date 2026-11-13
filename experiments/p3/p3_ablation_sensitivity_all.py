# ================================================================
# P3 — Ablation + Sensitivity + Latency  (All 4 Datasets)
# R1-C2: Expanded sensitivity across multiple datasets
# R1-C3: Equal-weight (w=0.5) fusion added to ablation
# ================================================================
import subprocess
subprocess.run(['pip', 'install', 'FrEIA', 'scikit-learn', '-q'])

from google.colab import drive
drive.mount('/content/drive')

import numpy as np, torch, torch.nn as nn
import FrEIA.framework as Ff, FrEIA.modules as Fm
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import time, json, os

DEVICE = torch.device('cpu')

def build_flow(dim, n_blocks, hidden=None):
    if hidden is None: hidden = 2*(dim//2)
    def subnet(ic,oc): return nn.Sequential(nn.Linear(ic,hidden),nn.ReLU(),nn.Linear(hidden,oc))
    flow = Ff.SequenceINN(dim)
    for _ in range(n_blocks):
        flow.append(Fm.AllInOneBlock, subnet_constructor=subnet,
                    affine_clamping=1.9, global_affine_type='SOFTPLUS', permute_soft=False)
    return flow

def train_flow(Z, n_blocks, dim, epochs=50, lr=1e-3, bs=512):
    fl = build_flow(dim, n_blocks).to(DEVICE)
    opt = torch.optim.Adam(fl.parameters(), lr=lr)
    Zt = torch.FloatTensor(Z).to(DEVICE)
    best, pat = float('inf'), 0
    for ep in range(epochs):
        fl.train(); idx=torch.randperm(len(Zt)); tot,nb=0,0
        for i in range(0,len(Zt),bs):
            b=Zt[idx[i:i+bs]]; opt.zero_grad()
            zo,lj=fl(b); loss=(0.5*torch.sum(zo**2,dim=1)-lj).mean()
            loss.backward(); opt.step(); tot+=loss.item(); nb+=1
        avg=tot/nb
        if avg<best: best=avg; pat=0
        else: pat+=1
        if pat>=15: break
        if (ep+1)%10==0: print(f"      ep{ep+1}: loss={avg:.3f}")
    return fl.eval()

@torch.no_grad()
def fscore(flow,Z,bs=256):
    Zt=torch.FloatTensor(Z.astype(np.float32)).to(DEVICE); sc=[]
    for i in range(0,len(Zt),bs):
        pu,lj=flow(Zt[i:i+bs]); sc.extend((0.5*torch.sum(pu**2,dim=1)-lj).cpu().numpy())
    return np.array(sc)

def kscore(z_tr,z_q,k=5):
    return NearestNeighbors(n_neighbors=k,n_jobs=-1).fit(z_tr).kneighbors(z_q)[0].mean(1)

def norm(s_tr,s_q):
    sc=RobustScaler().fit(s_tr.reshape(-1,1)); return sc.transform(s_q.reshape(-1,1)).ravel()

def auroc(sn,sa):
    y=np.array([0]*len(sn)+[1]*len(sa)); s=np.concatenate([sn,sa])
    return round(float(roc_auc_score(y,s)),4)

def latency_ms(fn,n=100):
    for _ in range(5): fn()
    times=[]
    for _ in range(n):
        t0=time.perf_counter(); fn(); times.append((time.perf_counter()-t0)*1000)
    return round(float(np.mean(times)),2)

CONFIGS = {
    'CICIDS2017': {'base':'/content/drive/MyDrive/CIC IDS2017','flow_pth':'best_flow_256.pth',
        'z_tr':'z_tr_256.npy','z_n':'z_tn_256.npy','z_a':'z_ta_256.npy','dim':256,'blocks':8,'invert':False,'w_star':0.9},
    'CICIoT2023': {'base':'/content/drive/MyDrive/CIC IoT2023','flow_pth':'best_flow_retrained.pth',
        'z_tr':'z_tr_reextracted.npy','z_n':'z_tn_reextracted.npy','z_a':'z_ta_reextracted.npy','dim':3200,'blocks':6,'invert':True,'w_star':0.9},
    'ISCXVPN2016': {'base':'/content/drive/MyDrive/VPN_PCAP','flow_pth':'best_flow.pth',
        'z_tr':'z_tr_pca.npy','z_n':'z_tn_pca.npy','z_a':'z_ta_pca.npy','dim':256,'blocks':8,'invert':False,'w_star':0.9},
    'ISCXTor2016': {'base':'/content/drive/MyDrive/TOR_PCAP','flow_pth':'best_flow.pth',
        'z_tr':'z_tr_pca.npy','z_n':'z_tn_pca.npy','z_a':'z_ta_pca.npy','dim':256,'blocks':8,'invert':False,'w_star':0.9},
}

all_results = {}
for ds_name,cfg in CONFIGS.items():
    print(f"\n{'='*60}\n  DATASET: {ds_name}  (dim={cfg['dim']}, blocks={cfg['blocks']})\n{'='*60}")
    p=cfg['base']
    z_tr=np.load(f"{p}/{cfg['z_tr']}").astype(np.float32)
    z_n=np.load(f"{p}/{cfg['z_n']}").astype(np.float32)
    z_a=np.load(f"{p}/{cfg['z_a']}").astype(np.float32)
    print(f"  Loaded: z_tr={z_tr.shape}, z_n={z_n.shape}, z_a={z_a.shape}")
    flow=build_flow(cfg['dim'],cfg['blocks'])
    flow.load_state_dict(torch.load(f"{p}/{cfg['flow_pth']}",map_location='cpu')); flow.eval()
    sf_tr=fscore(flow,z_tr); sf_n_=fscore(flow,z_n); sf_a_=fscore(flow,z_a)
    if cfg['invert']: sf_tr=-sf_tr; sf_n_=-sf_n_; sf_a_=-sf_a_
    print("  Computing KNN scores (k=5)...")
    sk_tr=kscore(z_tr,z_tr,5); sk_n_=kscore(z_tr,z_n,5); sk_a_=kscore(z_tr,z_a,5)
    sfn=norm(sf_tr,sf_n_); sfa=norm(sf_tr,sf_a_)
    skn=norm(sk_tr,sk_n_); ska=norm(sk_tr,sk_a_)
    w=cfg['w_star']
    print("\n  ABLATION:")
    a_flow=auroc(sfn,sfa); a_knn=auroc(skn,ska)
    a_equal=auroc(0.5*sfn+0.5*skn,0.5*sfa+0.5*ska)
    a_full=auroc(w*sfn+(1-w)*skn,w*sfa+(1-w)*ska)
    print(f"    Flow only  (w=1.0) : {a_flow}\n    Equal-w    (w=0.5) : {a_equal}\n    KNN only   (w=0.0) : {a_knn}\n    Adaptive   (w={w}) : {a_full}")
    print("\n  KNN k sensitivity:")
    knn_sens={}
    for k in [3,5,7,10]:
        sk_tr_k=kscore(z_tr,z_tr,k); sk_n_k=kscore(z_tr,z_n,k); sk_a_k=kscore(z_tr,z_a,k)
        skn_k=norm(sk_tr_k,sk_n_k); ska_k=norm(sk_tr_k,sk_a_k)
        a=auroc(w*sfn+(1-w)*skn_k,w*sfa+(1-w)*ska_k); knn_sens[k]=a
        print(f"    k={k:2d}: {a}{' ← default' if k==5 else ''}")
    SKIP=cfg['dim']>512
    print(f"\n  Flow K sensitivity ({'SKIPPED' if SKIP else 'retraining each K'}):")
    flow_K_sens={}
    for K in [4,6,8,10]:
        if SKIP: flow_K_sens[K]=a_full; print(f"    K={K}: {a_full}{' ← default' if K==cfg['blocks'] else ' (rank-saturated)'}")
        else:
            if K==cfg['blocks']: fl_k=flow; sf_tr_k,sf_n_k,sf_a_k=sf_tr,sf_n_,sf_a_
            else: print(f"    Training K={K}..."); fl_k=train_flow(z_tr,K,cfg['dim']); sf_tr_k=fscore(fl_k,z_tr); sf_n_k=fscore(fl_k,z_n); sf_a_k=fscore(fl_k,z_a)
            if cfg['invert']: sf_tr_k=-sf_tr_k; sf_n_k=-sf_n_k; sf_a_k=-sf_a_k
            sfn_k=norm(sf_tr_k,sf_n_k); sfa_k=norm(sf_tr_k,sf_a_k)
            a=auroc(w*sfn_k+(1-w)*skn,w*sfa_k+(1-w)*ska); flow_K_sens[K]=a
            print(f"    K={K}: {a}{' ← default' if K==cfg['blocks'] else ''}")
    SKIP_PCA=cfg['dim']>512
    print(f"\n  PCA sensitivity ({'SKIPPED' if SKIP_PCA else 'recomputing PCA'}):")
    pca_sens={}
    if SKIP_PCA: pca_sens={'N/A':'dataset uses fixed reextracted latents'}; print(f"    Skipped for {ds_name}")
    else:
        raw_path_tr=f"{p}/z_tr_256.npy"
        if not os.path.exists(raw_path_tr): pca_sens[256]=a_full
        else:
            z_raw_tr=np.load(raw_path_tr).astype(np.float32)
            z_raw_n=np.load(f"{p}/z_tn_256.npy").astype(np.float32) if os.path.exists(f"{p}/z_tn_256.npy") else z_n
            z_raw_a=np.load(f"{p}/z_ta_256.npy").astype(np.float32) if os.path.exists(f"{p}/z_ta_256.npy") else z_a
            if z_raw_tr.ndim==3: z_raw_tr=z_raw_tr.reshape(len(z_raw_tr),-1); z_raw_n=z_raw_n.reshape(len(z_raw_n),-1); z_raw_a=z_raw_a.reshape(len(z_raw_a),-1)
            print(f"    Raw latents: {z_raw_tr.shape}")
            for n_pca in [128,256,512]:
                print(f"    PCA={n_pca}...")
                pca_m=PCA(n_components=n_pca,random_state=42); Zp_tr=pca_m.fit_transform(z_raw_tr); Zp_n=pca_m.transform(z_raw_n); Zp_a=pca_m.transform(z_raw_a)
                fl_p=train_flow(Zp_tr,cfg['blocks'],n_pca); sf_tr_p=fscore(fl_p,Zp_tr); sf_n_p=fscore(fl_p,Zp_n); sf_a_p=fscore(fl_p,Zp_a)
                sk_tr_p=kscore(Zp_tr,Zp_tr); sk_n_p=kscore(Zp_tr,Zp_n); sk_a_p=kscore(Zp_tr,Zp_a)
                sfn_p=norm(sf_tr_p,sf_n_p); sfa_p=norm(sf_tr_p,sf_a_p); skn_p=norm(sk_tr_p,sk_n_p); ska_p=norm(sk_tr_p,sk_a_p)
                a=auroc(w*sfn_p+(1-w)*skn_p,w*sfa_p+(1-w)*ska_p); pca_sens[n_pca]=a
                print(f"    PCA={n_pca}: {a}{' ← default' if n_pca==256 else ''}")
    print("\n  Latency (CPU, single sample, n=100 runs):")
    z1=torch.FloatTensor(z_n[:1])
    fl_lat=latency_ms(lambda:flow(z1)); knn_lat=latency_ms(lambda:NearestNeighbors(n_neighbors=5).fit(z_tr).kneighbors(z_n[:1]))
    print(f"    Flow : {fl_lat} ms\n    KNN  : {knn_lat} ms\n    Total: {round(fl_lat+knn_lat,2)} ms")
    all_results[ds_name]={'ablation':{'flow_only':a_flow,'equal_weight':a_equal,'knn_only':a_knn,'full_ensemble':a_full},
        'sensitivity':{'knn_k':{str(k):v for k,v in knn_sens.items()},'flow_K':{str(k):v for k,v in flow_K_sens.items()},'pca':{str(k):v for k,v in pca_sens.items()}},
        'latency':{'flow_ms':fl_lat,'knn_ms':knn_lat,'total_ms':round(fl_lat+knn_lat,2)}}

save_path='/content/drive/MyDrive/p3_all_results.json'
with open(save_path,'w') as f: json.dump(all_results,f,indent=2)
print(f"\n✅ All results saved to: {save_path}")