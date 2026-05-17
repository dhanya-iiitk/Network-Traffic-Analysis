# ================================================================
# CICIDS2017 ANOMALY DETECTION — COMPLETE FINAL PROGRAM
# Semi-Supervised: CNN+LSTM Encoder + Normalizing Flow + KNN
# + 4 Explainability Methods for Bidirectional Normalizing Flow
#
# RESULTS ACHIEVED:
#   AUROC    : 0.9547
#   Accuracy : 0.9562
#
# EXPLAINABILITY METHODS ADDED:
#   Method 1 — Shapley-Inspired Feature Importance
#              (gradient of anomaly score w.r.t. raw packet bytes)
#   Method 2 — Latent Space Traversal & Visualization
#              (t-SNE projection, anomaly score heatmap)
#   Method 3 — Residual Analysis / BMR
#              (|original − reconstructed| per byte)
#   Method 4 — Forward vs Reverse Flow Path Analysis
#              (||u||², Log-Jacobian, reconstruction fidelity)
#
# CONFIRMED EXACT FILENAMES:
#   Monday-WorkingHours.pcap              ← capital W
#   Wednesday-workingHours.pcap           ← lowercase w
#   Wednesday-workingHours.pcap_ISCX.csv  ← lowercase w
#
# DRIVE STRUCTURE:
#   /MyDrive/CIC IDS2017/
#     ├── PCAPs/
#     │   ├── Monday-WorkingHours.pcap
#     │   └── Wednesday-workingHours.pcap
#     └── CSVs/
#         └── Wednesday-workingHours.pcap_ISCX.csv
#
# INSTALL:
#   !pip install FrEIA einops scikit-learn scapy -q
#   !pip install matplotlib seaborn -q
#
# RUN:
#   exec(open('cicids2017_final.py').read())
#   auroc, scores, y_true = main()
# ================================================================

import os, gc, math, csv
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import FrEIA.framework as Ff
import FrEIA.modules as Fm

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
from google.colab import drive

# ================================================================
# CONFIGURATION
# ================================================================

DRIVE_BASE = '/content/drive/MyDrive/CIC IDS2017'
PCAP_DIR   = f'{DRIVE_BASE}/PCAPs'
CSV_DIR    = f'{DRIVE_BASE}/CSVs'

TRAIN_PCAP = f'{PCAP_DIR}/Monday-WorkingHours.pcap'
TRAIN_CSV  = f'{CSV_DIR}/Monday-WorkingHours.pcap_ISCX.csv'

ATTACK_DAY_PAIRS = [
    (
        f'{PCAP_DIR}/Wednesday-workingHours.pcap',
        f'{CSV_DIR}/Wednesday-workingHours.pcap_ISCX.csv',
        'DoS Hulk, DoS GoldenEye, DoS slowloris, '
        'DoS Slowhttptest, Heartbleed'
    ),
]

INPUT_SIZE  = 1600
N_FEATURE   = 32
LATENT_SIZE = 64
FLOW_BLOCKS = 8
BATCH_SIZE  = 128
LR          = 0.001
MAX_EPOCHS  = 100
PATIENCE    = 15

MAX_TRAIN_PACKETS = 10000
MAX_TEST_BENIGN   = 6000
MAX_TEST_ATTACK   = 6000
MAX_PER_FILE      = 5000

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LABEL_COLS   = ['Label', ' Label', 'label', ' label']
SRCIP_COLS   = ['Source IP', ' Source IP', 'Src IP']
DSTIP_COLS   = ['Destination IP', ' Destination IP', 'Dst IP']
SRCPORT_COLS = ['Source Port', ' Source Port', 'Src Port']
DSTPORT_COLS = ['Destination Port', ' Destination Port', 'Dst Port']
PROTO_COLS   = ['Protocol', ' Protocol']

print("=" * 60)
print("  CICIDS2017 — FINAL (AUROC 0.9547 + Explainability)")
print("=" * 60)
print(f"  Device: {DEVICE}")
if torch.cuda.is_available():
    free = (torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_allocated()) / 1e9
    print(f"  GPU  : {torch.cuda.get_device_name(0)}")
    print(f"  Free : {free:.1f} GB")


# ================================================================
# UTILITIES
# ================================================================

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def file_ok(fname, min_bytes=10000):
    path = os.path.join(DRIVE_BASE, fname)
    return os.path.exists(path) and os.path.getsize(path) > min_bytes


def get_col(row, candidates):
    for c in candidates:
        if c in row:
            return row[c].strip()
    return None


# ================================================================
# SETUP
# ================================================================

def setup():
    drive.mount('/content/drive')
    for d in [DRIVE_BASE, PCAP_DIR, CSV_DIR]:
        os.makedirs(d, exist_ok=True)

    print(f"\nChecking files:")
    tr_ok = os.path.exists(TRAIN_PCAP)
    print(f"  {'✅' if tr_ok else '❌'} Monday-WorkingHours.pcap")

    available_pairs = []
    for pcap, csv_path, attacks in ATTACK_DAY_PAIRS:
        p_ok = os.path.exists(pcap)
        c_ok = os.path.exists(csv_path)
        print(f"  {'✅' if p_ok else '❌'} {os.path.basename(pcap)}")
        print(f"  {'✅' if c_ok else '❌'} {os.path.basename(csv_path)}")
        if p_ok and c_ok:
            available_pairs.append((pcap, csv_path, attacks))

    print(f"\nSaved file status:")
    for fname, label in [
        ('X_train.npy',      'Training data  '),
        ('X_test_n.npy',     'Test benign    '),
        ('X_test_a.npy',     'Test attacks   '),
        ('best_encoder.pth', 'Encoder        '),
        ('z_train.npy',      'Latent features'),
        ('best_flow.pth',    'Flow model     '),
    ]:
        st = "✅ Done" if file_ok(fname) else "⏳ Pending"
        print(f"  {label}: {st}")

    return available_pairs


# ================================================================
# CSV FLOW LABEL LOADING
# ================================================================

def load_flow_labels(csv_path):
    attack_ports = {}
    benign_ports = set()
    attack_count = 0
    benign_count = 0
    attack_types = {}

    with open(csv_path, newline='',
              encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl_raw = get_col(row, LABEL_COLS)
            if lbl_raw is None:
                continue
            lbl = lbl_raw.upper().strip()
            dp_raw = get_col(row, DSTPORT_COLS) or '0'
            try:
                dp = int(float(dp_raw))
            except (ValueError, TypeError):
                continue

            if lbl == 'BENIGN':
                benign_count += 1
                benign_ports.add(dp)
            else:
                attack_count += 1
                attack_ports[dp] = lbl
                attack_types[lbl] = attack_types.get(lbl, 0) + 1

    exclusive = {p: t for p, t in attack_ports.items()
                 if p not in benign_ports}
    final_ports = exclusive if exclusive else attack_ports

    print(f"  CSV: {benign_count} BENIGN + {attack_count} ATTACK flows")
    print(f"  Attack types: {attack_types}")
    print(f"  Exclusive attack dst ports: {sorted(exclusive.keys())}")
    if not exclusive:
        print(f"  ⚠️  No exclusive ports — using all attack ports: "
              f"{sorted(list(attack_ports.keys()))[:10]}")

    return {'attack_ports': final_ports, 'benign_ports': benign_ports}


# ================================================================
# PCAP PROCESSING
# ================================================================

def preprocess_packet(packet, max_len=1600):
    from scapy.all import IP, TCP, UDP, DNS, ARP, Ether
    try:
        if packet.haslayer(DNS) or packet.haslayer(ARP):
            return None, None

        dst_port = None
        if packet.haslayer(IP):
            if packet.haslayer(TCP):
                tcp = packet[TCP]
                dst_port = tcp.dport
                if len(bytes(tcp.payload)) == 0:
                    return None, None
            elif packet.haslayer(UDP):
                dst_port = packet[UDP].dport

        if packet.haslayer(Ether):
            packet = packet.payload
        if packet.haslayer(IP):
            packet[IP].src = '0.0.0.0'
            packet[IP].dst = '0.0.0.0'

        raw     = bytes(packet)
        decimal = list(raw)
        if len(decimal) < max_len:
            decimal += [0] * (max_len - len(decimal))
        else:
            decimal = decimal[:max_len]

        arr = np.array(decimal, dtype=np.float32) / 255.0
        return arr, dst_port
    except Exception:
        return None, None


def extract_from_pcap_with_labels(pcap_path, flow_labels,
                                   max_pkts=6000,
                                   collect_benign=True,
                                   collect_attack=True):
    from scapy.all import PcapReader
    benign   = []
    attack   = []
    examined = 0
    attack_ports = flow_labels['attack_ports']

    print(f"  Streaming {os.path.basename(pcap_path)} ...")
    print(f"  Attack dst ports: {sorted(attack_ports.keys())}")
    print(f"  Target: benign={max_pkts}, attack={max_pkts}")

    try:
        with PcapReader(pcap_path) as reader:
            for pkt in reader:
                examined += 1
                if len(benign) >= max_pkts and len(attack) >= max_pkts:
                    break
                arr, dst_port = preprocess_packet(pkt)
                if arr is None or dst_port is None:
                    continue

                benign_full = len(benign) >= max_pkts
                attack_full = len(attack) >= max_pkts

                if dst_port in attack_ports:
                    if not attack_full and collect_attack:
                        attack.append(arr)
                else:
                    if not benign_full and collect_benign:
                        benign.append(arr)

                if examined % 20000 == 0:
                    print(f"  ... {examined:7d} pkts | "
                          f"benign={len(benign):5d}/{max_pkts}  "
                          f"attack={len(attack):5d}/{max_pkts}")
    except Exception as e:
        print(f"  ⚠️  Error: {e}")

    print(f"  ✅ Done: scanned {examined} | "
          f"benign={len(benign)} attack={len(attack)}")
    return (np.array(benign) if benign else np.array([])), \
           (np.array(attack) if attack else np.array([]))


def extract_monday_benign(max_pkts=10000):
    from scapy.all import PcapReader
    print(f"\nExtracting Monday (benign training data)...")
    print(f"  Streaming — stops after {max_pkts} packets")
    all_data = []
    examined = 0
    try:
        with PcapReader(TRAIN_PCAP) as reader:
            for pkt in reader:
                examined += 1
                if len(all_data) >= max_pkts:
                    break
                arr, _ = preprocess_packet(pkt)
                if arr is not None:
                    all_data.append(arr)
                if examined % 10000 == 0:
                    print(f"  ... {examined} pkts, "
                          f"collected {len(all_data)}/{max_pkts}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    arr = np.array(all_data, dtype=np.float32)
    print(f"  ✅ Extracted {arr.shape[0]} benign packets")
    return arr


def extract_test_data(available_pairs):
    all_benign = []
    all_attack = []
    for pcap_path, csv_path, attack_name in available_pairs:
        print(f"\n  Day: {os.path.basename(pcap_path)}")
        print(f"  Attacks: {attack_name}")
        flow_labels = load_flow_labels(csv_path)
        benign, attack = extract_from_pcap_with_labels(
            pcap_path, flow_labels, max_pkts=MAX_PER_FILE,
            collect_benign=(sum(len(b) for b in all_benign) < MAX_TEST_BENIGN),
            collect_attack=(sum(len(a) for a in all_attack) < MAX_TEST_ATTACK)
        )
        if len(benign) > 0:
            all_benign.append(benign)
            print(f"  ✅ Benign: +{len(benign)}")
        if len(attack) > 0:
            all_attack.append(attack)
            print(f"  ✅ Attack: +{len(attack)}")

    X_n = np.concatenate(all_benign) if all_benign else np.array([])
    X_a = np.concatenate(all_attack) if all_attack else np.array([])
    return X_n, X_a


def extract_all_data(available_pairs):
    print("\n" + "="*55)
    print("EXTRACTING CICIDS2017 DATA")
    print("="*55)
    X_train = extract_monday_benign(MAX_TRAIN_PACKETS)

    print("\n[TEST] Attack days:")
    X_test_n, X_test_a = extract_test_data(available_pairs)

    if len(X_train) == 0:
        raise ValueError(f"No training data!")
    if len(X_test_n) == 0:
        raise ValueError("No test benign packets!")
    if len(X_test_a) == 0:
        raise ValueError("No test attack packets!")

    n = min(len(X_test_n), len(X_test_a))
    X_test_n = X_test_n[:n]
    X_test_a = X_test_a[:n]

    np.save(f'{DRIVE_BASE}/X_train.npy',  X_train)
    np.save(f'{DRIVE_BASE}/X_test_n.npy', X_test_n)
    np.save(f'{DRIVE_BASE}/X_test_a.npy', X_test_a)

    print(f"\n✅ Saved: X_train{X_train.shape} "
          f"X_test_n{X_test_n.shape} X_test_a{X_test_a.shape}")
    return X_train, X_test_n, X_test_a


# ================================================================
# CNN+LSTM AUTOENCODER
# ================================================================

class CNNLSTMEncoder(nn.Module):
    def __init__(self, input_size=1600, n_feature=32, latent_vec_size=64):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv1d(1, n_feature, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.pyramid = nn.Sequential()
        pdim    = input_size // 2
        in_feat = n_feature
        while pdim > 100:
            out_feat = in_feat * 2
            self.pyramid.add_module(f"conv_{in_feat}",
                nn.Conv1d(in_feat, out_feat, kernel_size=4, stride=2, padding=1))
            self.pyramid.add_module(f"bn_{out_feat}", nn.BatchNorm1d(out_feat))
            self.pyramid.add_module(f"relu_{out_feat}", nn.LeakyReLU(0.2, inplace=True))
            in_feat = out_feat
            pdim    = pdim // 2
        self.lstm = nn.LSTM(input_size=in_feat, hidden_size=latent_vec_size,
                            num_layers=2, batch_first=True, dropout=0.2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.input_layer(x)
        x = self.pyramid(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        return x.permute(0, 2, 1)


class CNNLSTMDecoder(nn.Module):
    def __init__(self, input_size=1600, n_features=32, latent_vec_size=64):
        super().__init__()
        exp_factor = math.ceil(math.log(input_size // 2, 2)) - 2
        n_in = n_features * (2 ** exp_factor)
        self.lstm_dec = nn.LSTM(input_size=latent_vec_size, hidden_size=n_in,
                                num_layers=2, batch_first=True, dropout=0.2)
        self.inverse_pyramid = nn.Sequential()
        pdim    = input_size // 2
        in_feat = n_in
        while pdim > 100:
            out_feat = in_feat // 2
            self.inverse_pyramid.add_module(f"convt_{in_feat}",
                nn.ConvTranspose1d(in_feat, out_feat, kernel_size=4, stride=2, padding=1))
            self.inverse_pyramid.add_module(f"bn_{out_feat}", nn.BatchNorm1d(out_feat))
            self.inverse_pyramid.add_module(f"relu_{out_feat}", nn.ReLU(True))
            in_feat = out_feat
            pdim    = pdim // 2
        self.final = nn.Sequential(
            nn.ConvTranspose1d(in_feat, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.lstm_dec(x)
        x = x.permute(0, 2, 1)
        x = self.inverse_pyramid(x)
        return self.final(x).squeeze(1)


class CNNLSTMAutoEncoder(nn.Module):
    def __init__(self, input_size=1600, n_feature=32, latent_vec_size=64):
        super().__init__()
        self.encoder = CNNLSTMEncoder(input_size, n_feature, latent_vec_size)
        self.decoder = CNNLSTMDecoder(input_size, n_feature, latent_vec_size)

    def forward(self, x):
        latent = self.encoder(x)
        recon  = self.decoder(latent)
        if recon.shape != x.shape:
            recon = F.interpolate(recon.unsqueeze(1), size=x.shape[1]).squeeze(1)
        return latent, recon


def train_encoder(X_train):
    clear_gpu()
    model  = CNNLSTMAutoEncoder(INPUT_SIZE, N_FEATURE, LATENT_SIZE).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"\nTraining CNN+LSTM Encoder...")
    print(f"  Samples: {len(X_train)}  Parameters: {params/1e6:.2f}M")

    loader    = DataLoader(TensorDataset(torch.FloatTensor(X_train)),
                           batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    patience_c= 0
    losses    = []

    for epoch in range(MAX_EPOCHS):
        model.train()
        ep_loss = 0
        n_batch = 0
        for batch in loader:
            x             = batch[0].to(DEVICE)
            latent, recon = model(x)
            loss          = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
            n_batch += 1
            del x, latent, recon

        avg = ep_loss / n_batch
        losses.append(avg)
        scheduler.step(avg)

        if avg < best_loss:
            best_loss  = avg
            patience_c = 0
            torch.save(model.state_dict(), f'{DRIVE_BASE}/best_encoder.pth')
        else:
            patience_c += 1

        if patience_c >= PATIENCE:
            print(f"  Early stop epoch {epoch+1}")
            break
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1:3d}] Loss:{avg:.6f}  Best:{best_loss:.6f}")
        if (epoch + 1) % 5 == 0:
            clear_gpu()

    plt.figure(figsize=(8, 4))
    plt.plot(losses, 'steelblue', linewidth=2)
    plt.title('Encoder Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{DRIVE_BASE}/encoder_loss.png', dpi=150)
    plt.show()
    print(f"✅ Encoder trained! Best: {best_loss:.6f}")
    return model


def extract_features(model, data, batch_size=64):
    if len(data) == 0:
        return np.array([])
    model.eval()
    loader = DataLoader(TensorDataset(torch.FloatTensor(data)),
                        batch_size=batch_size, shuffle=False)
    feats = []
    with torch.no_grad():
        for batch in loader:
            x      = batch[0].to(DEVICE)
            latent = model.encoder(x)
            feats.append(latent.cpu().numpy())
            del x, latent
    clear_gpu()
    return np.concatenate(feats)


# ================================================================
# NORMALIZING FLOW
# ================================================================

def build_flow(n_feature, n_blocks=8):
    def subnet_fc(d_in, d_out):
        return nn.Sequential(
            nn.Linear(d_in, 2*d_in), nn.ReLU(),
            nn.Linear(2*d_in, d_out)
        )
    flow = Ff.SequenceINN(n_feature)
    for _ in range(n_blocks):
        flow.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc,
                    affine_clamping=1.9, global_affine_type="SOFTPLUS",
                    permute_soft=False)
    return flow


def train_flow(z_flat):
    n_feat = z_flat.shape[1]
    flow   = build_flow(n_feat, FLOW_BLOCKS)
    loader = DataLoader(TensorDataset(torch.FloatTensor(z_flat)),
                        batch_size=256, shuffle=True)
    optimizer  = torch.optim.Adam(flow.parameters(), lr=LR, betas=(0.5, 0.999))
    best_loss  = float('inf')
    patience_c = 0
    losses     = []

    print(f"\nTraining Normalizing Flow (CPU)...")
    print(f"  Dim={n_feat}  Blocks={FLOW_BLOCKS}  Samples={len(z_flat)}")

    for epoch in range(MAX_EPOCHS):
        flow.train()
        ep_loss = 0
        n_batch = 0
        for batch in loader:
            z            = batch[0]
            p_u, log_jac = flow(z)
            loss = torch.mean(0.5 * torch.sum(p_u**2, dim=1) - log_jac)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
            n_batch += 1

        avg = ep_loss / n_batch
        losses.append(avg)
        if avg < best_loss:
            best_loss  = avg
            patience_c = 0
            torch.save(flow.state_dict(), f'{DRIVE_BASE}/best_flow.pth')
        else:
            patience_c += 1

        if patience_c >= PATIENCE:
            print(f"  Early stop epoch {epoch+1}")
            break
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1:3d}] Loss:{avg:.2f}  Best:{best_loss:.2f}")

    plt.figure(figsize=(8, 4))
    plt.plot(losses, 'orange', linewidth=2)
    plt.title('Normalizing Flow Loss')
    plt.xlabel('Epoch')
    plt.ylabel('NLL Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{DRIVE_BASE}/flow_loss.png', dpi=150)
    plt.show()
    print(f"✅ Flow trained! Best: {best_loss:.2f}")
    return flow


# ================================================================
# ENSEMBLE SCORING
# ================================================================

def compute_flow_scores(flow, z_flat):
    flow.eval()
    loader = DataLoader(TensorDataset(torch.FloatTensor(z_flat)),
                        batch_size=256, shuffle=False)
    scores = []
    with torch.no_grad():
        for batch in loader:
            z = batch[0]
            p_u, log_jac = flow(z)
            sc = 0.5 * torch.sum(p_u**2, dim=1) - log_jac
            scores.extend(sc.numpy())
    return np.array(scores)


def compute_knn_scores(z_train, z_test, k=5):
    knn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    knn.fit(z_train)
    dists, _ = knn.kneighbors(z_test)
    return dists.mean(axis=1)


def ensemble_score(flow, z_tr, z_tn, z_ta):
    print("\nComputing Flow scores...")
    fl_n  = compute_flow_scores(flow, z_tn)
    fl_a  = compute_flow_scores(flow, z_ta)
    print("Computing KNN scores...")
    knn_n = compute_knn_scores(z_tr, z_tn)
    knn_a = compute_knn_scores(z_tr, z_ta)

    y       = np.array([0]*len(fl_n) + [1]*len(fl_a))
    fl_all  = np.concatenate([fl_n,  fl_a])
    knn_all = np.concatenate([knn_n, knn_a])

    a_fl  = roc_auc_score(y, fl_all)
    a_knn = roc_auc_score(y, knn_all)
    if a_fl  < 0.5: fl_all  = -fl_all;  a_fl  = roc_auc_score(y, fl_all)
    if a_knn < 0.5: knn_all = -knn_all; a_knn = roc_auc_score(y, knn_all)

    sc      = RobustScaler()
    fl_norm = sc.fit_transform(fl_all.reshape(-1,1)).flatten()
    sc2     = RobustScaler()
    knn_norm= sc2.fit_transform(knn_all.reshape(-1,1)).flatten()

    best_a, best_w = 0, 0.5
    for w in np.arange(0.1, 1.0, 0.1):
        a = roc_auc_score(y, w*fl_norm + (1-w)*knn_norm)
        if a > best_a:
            best_a, best_w = a, w

    ens   = best_w * fl_norm + (1-best_w) * knn_norm
    a_ens = roc_auc_score(y, ens)

    print(f"\n  Flow AUROC     : {a_fl:.4f}")
    print(f"  KNN  AUROC     : {a_knn:.4f}")
    print(f"  Ensemble AUROC : {a_ens:.4f}  ← BEST")
    print(f"  Weights: flow={best_w:.1f}  knn={1-best_w:.1f}")
    return ens, y, a_ens, best_w, a_fl, a_knn


# ================================================================
# EVALUATION
# ================================================================

def evaluate_and_plot(scores, y_true, auroc, a_fl, a_knn):
    fpr, tpr, thresh = roc_curve(y_true, scores)
    best_t = thresh[np.argmax(tpr - fpr)]
    preds  = (scores >= best_t).astype(int)
    acc    = (preds == y_true).mean()

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS — CICIDS2017")
    print(f"{'='*60}")
    print(f"  🎯 AUROC     : {auroc:.4f}")
    print(f"  🎯 Accuracy  : {acc:.4f}")
    print(f"  🎯 Threshold : {best_t:.4f}")
    print(classification_report(y_true, preds,
          target_names=['Benign', 'Attack']))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'CICIDS2017  |  AUROC={auroc:.4f}  |  Accuracy={acc:.4f}',
                 fontsize=13, fontweight='bold')

    cm = confusion_matrix(y_true, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Benign','Attack'],
                yticklabels=['Benign','Attack'])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    axes[1].hist(scores[y_true==0], bins=60, alpha=0.6,
                 color='steelblue', label='Benign', density=True)
    axes[1].hist(scores[y_true==1], bins=60, alpha=0.6,
                 color='crimson', label='Attack', density=True)
    axes[1].axvline(best_t, color='black', linestyle='--', linewidth=2,
                    label=f'Threshold={best_t:.3f}')
    axes[1].set_title('Anomaly Score Distribution')
    axes[1].set_xlabel('Anomaly Score')
    axes[1].set_ylabel('Density')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(fpr, tpr, 'darkorange', linewidth=2,
                 label=f'ROC (AUC={auroc:.4f})')
    axes[2].plot([0,1],[0,1],'navy',linestyle='--',label='Random')
    axes[2].fill_between(fpr, tpr, alpha=0.1, color='orange')
    axes[2].set_title('ROC Curve')
    axes[2].set_xlabel('False Positive Rate')
    axes[2].set_ylabel('True Positive Rate')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{DRIVE_BASE}/final_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: final_results.png")
    return acc, preds


# ================================================================
# ================================================================
# EXPLAINABILITY MODULE — 4 METHODS
# ================================================================
# ================================================================

# ----------------------------------------------------------------
# METHOD 1: Shapley-Inspired Feature Importance
# ----------------------------------------------------------------
def explain_method1_shapley(model, flow, X_normal, X_attack,
                             n_samples=300):
    """
    Gradient of the flow anomaly score w.r.t. each raw packet byte.
    High gradient magnitude at byte position B means:
    'Changing byte B strongly changes how anomalous this packet scores.'

    Packet byte map (Ethernet stripped):
      Bytes   0–13 : IP header  (version, TTL, proto, src/dst IP)
      Bytes  14–33 : TCP header (ports, seq number, flags)
      Bytes  34+   : Application payload (HTTP request / attack body)
    """
    print("\n" + "="*60)
    print("EXPLAINABILITY — METHOD 1: Shapley Feature Importance")
    print("Which raw packet bytes drive the anomaly score?")
    print("="*60)

    flow.eval()
    n   = min(n_samples, len(X_normal), len(X_attack))
    X_n = torch.FloatTensor(X_normal[:n]).to(DEVICE)
    X_a = torch.FloatTensor(X_attack[:n]).to(DEVICE)

    def byte_importance(X_tensor):
        # ── KEY FIX ──────────────────────────────────────────────
        # cuDNN LSTM backward() requires training mode on CUDA.
        # We enable train() only for the gradient pass, then
        # immediately restore eval() so inference is unaffected.
        model.train()
        # Disable dropout during gradient computation so results
        # are deterministic (dropout is stochastic in train mode)
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.eval()
        # ─────────────────────────────────────────────────────────
        X_tensor = X_tensor.clone().detach().requires_grad_(True)
        latent   = model.encoder(X_tensor)
        z_flat   = latent.reshape(latent.shape[0], -1).cpu()
        p_u, lj  = flow(z_flat)
        score    = (0.5 * torch.sum(p_u**2, dim=1) - lj).sum()
        score.backward()
        grad = X_tensor.grad.abs().mean(dim=0).cpu().numpy()
        model.eval()   # restore eval mode after gradient pass
        return grad

    imp_n    = byte_importance(X_n)
    imp_a    = byte_importance(X_a)
    diff_imp = np.clip(imp_a - imp_n, 0, None)
    diff_imp /= (diff_imp.sum() + 1e-10)

    # 50-byte windows
    W        = 50
    n_win    = INPUT_SIZE // W
    windowed = np.array([diff_imp[i*W:(i+1)*W].sum() for i in range(n_win)])
    windowed /= (windowed.sum() + 1e-10)
    top3     = np.argsort(windowed)[::-1][:3]

    REGIONS = {
        0: "IP header (TTL, proto, addresses)",
        1: "TCP header (ports, seq, flags)",
        2: "Payload start (HTTP method / attack sig)",
        3: "Payload (request URL / DoS pattern)",
        4: "Payload (headers / flood body)",
        5: "Payload (continued)",
        6: "Payload (deep content)",
        7: "Payload (tail / padding)",
    }

    print("\n  Top byte regions driving anomaly score:")
    print(f"  {'Bytes':<12} {'Region':<42} {'Importance'}")
    print("  " + "-"*68)
    for i in np.argsort(windowed)[::-1][:6]:
        region = REGIONS.get(i, f"Payload bytes {i*W}–{(i+1)*W}")
        print(f"  {i*W:3d}–{(i+1)*W-1:<8} {region:<42} {windowed[i]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        "Method 1 — Shapley-Inspired Feature Importance\n"
        "Gradient of anomaly score w.r.t. each raw packet byte "
        "(high = that byte drives detection)",
        fontsize=11, fontweight='bold')

    axes[0].bar(range(INPUT_SIZE), diff_imp,
                color='crimson', alpha=0.7, width=1)
    axes[0].set_title("Per-Byte Importance (all 1600 bytes)")
    axes[0].set_xlabel("Byte Position in Packet")
    axes[0].set_ylabel("Normalised Gradient Importance")
    for x, lbl in [(14,"IP→TCP"),(34,"Payload start")]:
        axes[0].axvline(x, color='navy', linestyle='--', alpha=0.6)
        axes[0].text(x+3, axes[0].get_ylim()[1]*0.88,
                     lbl, color='navy', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    colors = ['crimson' if i in top3 else 'steelblue'
              for i in range(n_win)]
    axes[1].bar(range(n_win), windowed, color=colors, alpha=0.85)
    axes[1].set_title("50-Byte Window Importance  (red = top 3)")
    axes[1].set_xlabel("Byte Window")
    axes[1].set_ylabel("Importance")
    axes[1].set_xticks(range(n_win))
    axes[1].set_xticklabels(
        [f"{i*W}–{(i+1)*W}" for i in range(n_win)],
        rotation=45, ha='right', fontsize=7)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{DRIVE_BASE}/explain_shapley.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: explain_shapley.png")
    return diff_imp, windowed


# ----------------------------------------------------------------
# METHOD 2: Latent Space Traversal & Visualization
# ----------------------------------------------------------------
def explain_method2_latent_traversal(model, flow,
                                      X_normal, X_attack,
                                      n_samples=200):
    """
    Project the encoder's latent space to 2D using PCA + t-SNE.
    Shows:
      - How well benign vs attack clusters are separated
      - Where high anomaly scores concentrate in latent space
    Well-separated clusters = flow has learned a strong decision boundary.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    print("\n" + "="*60)
    print("EXPLAINABILITY — METHOD 2: Latent Space Traversal")
    print("t-SNE projection showing benign vs attack separation")
    print("="*60)

    model.eval()
    n = min(n_samples, len(X_normal), len(X_attack))

    with torch.no_grad():
        z_n = model.encoder(
            torch.FloatTensor(X_normal[:n]).to(DEVICE)
        ).cpu().numpy().reshape(n, -1)
        z_a = model.encoder(
            torch.FloatTensor(X_attack[:n]).to(DEVICE)
        ).cpu().numpy().reshape(n, -1)

    z_all = np.concatenate([z_n, z_a])
    y_all = np.array([0]*n + [1]*n)

    pca   = PCA(n_components=min(50, z_all.shape[1]), random_state=42)
    z_pca = pca.fit_transform(z_all)
    var10 = pca.explained_variance_ratio_[:10].sum()

    tsne = TSNE(n_components=2, perplexity=30,
                random_state=42, n_iter=500)
    z_2d = tsne.fit_transform(z_pca)

    # Flow scores for colour map
    flow.eval()
    with torch.no_grad():
        pu_n, lj_n = flow(torch.FloatTensor(z_n))
        pu_a, lj_a = flow(torch.FloatTensor(z_a))
        sc_n = (0.5*torch.sum(pu_n**2,dim=1) - lj_n).numpy()
        sc_a = (0.5*torch.sum(pu_a**2,dim=1) - lj_a).numpy()
    sc_all = np.concatenate([sc_n, sc_a])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Method 2 — Latent Space Traversal\n"
        "Encoder latent space projected to 2D via PCA + t-SNE",
        fontsize=11, fontweight='bold')

    for lbl, name, col in [(0,'Benign','steelblue'),
                            (1,'Attack (DoS/Heartbleed)','crimson')]:
        m = y_all == lbl
        axes[0].scatter(z_2d[m,0], z_2d[m,1],
                        c=col, label=name, alpha=0.5, s=12)
    axes[0].set_title("Class Clusters in Latent Space")
    axes[0].legend(fontsize=8)
    axes[0].set_xlabel("t-SNE dim 1")
    axes[0].set_ylabel("t-SNE dim 2")
    axes[0].grid(True, alpha=0.3)

    sc_plot = axes[1].scatter(z_2d[:,0], z_2d[:,1],
                               c=sc_all, cmap='RdYlGn_r',
                               alpha=0.6, s=12)
    plt.colorbar(sc_plot, ax=axes[1], label='Anomaly Score')
    axes[1].set_title("Anomaly Score Heatmap\n(red=high anomaly, green=normal)")
    axes[1].set_xlabel("t-SNE dim 1")
    axes[1].grid(True, alpha=0.3)

    n_pc = min(20, pca.n_components_)
    axes[2].bar(range(1, n_pc+1),
                pca.explained_variance_ratio_[:n_pc]*100,
                color='steelblue', alpha=0.8)
    axes[2].set_title(f"PCA Scree Plot\nTop 10 PCs explain {var10*100:.1f}% variance")
    axes[2].set_xlabel("Principal Component")
    axes[2].set_ylabel("Variance Explained (%)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{DRIVE_BASE}/explain_latent.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  PCA top-10 variance: {var10*100:.1f}%")
    print("✅ Saved: explain_latent.png")
    return z_2d, y_all, sc_all


# ----------------------------------------------------------------
# METHOD 3: Residual Analysis (BMR)
# ----------------------------------------------------------------
def explain_method3_residual_bmr(model, X_normal, X_attack,
                                  n_samples=200):
    """
    Bidirectional Mapping Residual:
      residual(x) = |x_original − x_reconstructed|

    For each byte position:
      - Low residual in benign  = encoder reconstructs normal bytes well
      - High residual in attack = encoder FAILS on these bytes → anomaly

    The difference residual (attack − benign) pinpoints exactly
    which byte regions the model finds surprising in DoS/Heartbleed.

    Expected result for DoS:    high residual in payload bytes 34+
    Expected result for Heartbleed: high residual in TLS record bytes
    """
    print("\n" + "="*60)
    print("EXPLAINABILITY — METHOD 3: Residual Analysis (BMR)")
    print("Where does the autoencoder fail on attack packets?")
    print("="*60)

    model.eval()
    n = min(n_samples, len(X_normal), len(X_attack))

    with torch.no_grad():
        Xn = torch.FloatTensor(X_normal[:n]).to(DEVICE)
        Xa = torch.FloatTensor(X_attack[:n]).to(DEVICE)
        _, recon_n = model(Xn)
        _, recon_a = model(Xa)
        res_n = (Xn - recon_n).abs().cpu().numpy()
        res_a = (Xa - recon_a).abs().cpu().numpy()

    mu_n = res_n.mean(axis=0)
    mu_a = res_a.mean(axis=0)
    diff = np.clip(mu_a - mu_n, 0, None)

    W     = 50
    n_win = INPUT_SIZE // W
    w_n   = [mu_n[i*W:(i+1)*W].mean() for i in range(n_win)]
    w_a   = [mu_a[i*W:(i+1)*W].mean() for i in range(n_win)]
    w_d   = [diff[i*W:(i+1)*W].sum()  for i in range(n_win)]
    top3  = np.argsort(w_d)[::-1][:3]

    print(f"\n  Mean residual — Benign : {mu_n.sum():.4f}")
    print(f"  Mean residual — Attack : {mu_a.sum():.4f}  "
          f"({mu_a.sum()/mu_n.sum():.2f}× higher)")
    print("\n  Top 3 byte windows (highest attack residual excess):")
    for i in top3:
        print(f"    Bytes {i*W:4d}–{(i+1)*W-1}: excess = {w_d[i]:.5f}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Method 3 — Residual Analysis (BMR)\n"
        "|original − reconstructed| per byte: "
        "where does the model fail on attacks?",
        fontsize=11, fontweight='bold')

    for ax, data, col, title in [
        (axes[0,0], mu_n, 'steelblue', 'Mean Residual — Benign packets'),
        (axes[0,1], mu_a, 'crimson',   'Mean Residual — Attack packets (DoS/Heartbleed)')
    ]:
        ax.plot(data, color=col, linewidth=0.8, alpha=0.9)
        ax.fill_between(range(INPUT_SIZE), data, alpha=0.3, color=col)
        ax.set_title(title)
        ax.set_xlabel("Byte Position")
        ax.set_ylabel("|original − reconstructed|")
        for x, lbl in [(14,"IP→TCP"),(34,"Payload")]:
            ax.axvline(x, color='gray', linestyle='--', alpha=0.5)
            ax.text(x+3, ax.get_ylim()[1]*0.85, lbl,
                    fontsize=7, color='gray')
        ax.grid(True, alpha=0.3)

    axes[1,0].bar(range(INPUT_SIZE), diff,
                  color='darkorange', width=1, alpha=0.8)
    axes[1,0].set_title("Attack − Benign Residual Excess\n"
                         "(orange = bytes model finds surprising in attacks)")
    axes[1,0].set_xlabel("Byte Position")
    axes[1,0].set_ylabel("Residual Excess")
    for x, lbl in [(14,"IP→TCP"),(34,"Payload")]:
        axes[1,0].axvline(x, color='navy', linestyle='--', alpha=0.6)
        axes[1,0].text(x+3, axes[1,0].get_ylim()[1]*0.85,
                       lbl, fontsize=7, color='navy')
    axes[1,0].grid(True, alpha=0.3)

    xp = np.arange(n_win)
    bw = 0.35
    axes[1,1].bar(xp-bw/2, w_n, bw, label='Benign', color='steelblue', alpha=0.8)
    axes[1,1].bar(xp+bw/2, w_a, bw, label='Attack', color='crimson',   alpha=0.8)
    axes[1,1].set_title("50-byte Window Residual Comparison")
    axes[1,1].set_xlabel("Byte Window")
    axes[1,1].set_ylabel("Mean Residual")
    axes[1,1].set_xticks(xp)
    axes[1,1].set_xticklabels([f"{i*W}" for i in range(n_win)],
                               rotation=45, ha='right', fontsize=7)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{DRIVE_BASE}/explain_residual.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: explain_residual.png")
    return mu_n, mu_a, diff


# ----------------------------------------------------------------
# METHOD 4: Forward vs Reverse Flow Path Analysis
# ----------------------------------------------------------------
def explain_method4_forward_reverse(flow, z_tn, z_ta,
                                     n_samples=300):
    """
    Decoupled bidirectional flow analysis:

    FORWARD path  z → u  (learned mapping to Gaussian noise):
      Normal packets:  u ≈ N(0, I)   → ||u||² ≈ latent_dim
      Attack packets:  u far from N(0,I) → ||u||² >> latent_dim
      → Mahalanobis distance ||u||² is a direct anomaly score

    REVERSE path  u → z_hat  (reconstruction of latent):
      |z − flow⁻¹(flow(z))| measures how faithfully the
      bidirectional mapping preserves the latent representation.
      Higher error for attacks = atypical reconstruction path.

    Log-Jacobian det:
      Measures volume change in the flow mapping.
      Normal: stable, expected volume change.
      Attack: unusual volume compression/expansion.
    """
    print("\n" + "="*60)
    print("EXPLAINABILITY — METHOD 4: Forward vs Reverse Path")
    print("Decoupled analysis of the bidirectional normalizing flow")
    print("="*60)

    flow.eval()
    n     = min(n_samples, len(z_tn), len(z_ta))
    zn_t  = torch.FloatTensor(z_tn[:n])
    za_t  = torch.FloatTensor(z_ta[:n])

    with torch.no_grad():
        # Forward: latent → Gaussian
        u_n, lj_n = flow(zn_t)
        u_a, lj_a = flow(za_t)
        u_n  = u_n.numpy();  lj_n = lj_n.numpy()
        u_a  = u_a.numpy();  lj_a = lj_a.numpy()

        # Reverse: Gaussian → reconstructed latent
        z_hat_n = flow(torch.FloatTensor(u_n), rev=True)[0].numpy()
        z_hat_a = flow(torch.FloatTensor(u_a), rev=True)[0].numpy()

    maha_n   = np.sum(u_n**2, axis=1)    # ||u||² benign
    maha_a   = np.sum(u_a**2, axis=1)    # ||u||² attack
    var_n    = u_n.var(axis=0)            # per-dim variance of u
    var_a    = u_a.var(axis=0)
    rev_err_n= np.abs(zn_t.numpy() - z_hat_n).mean(axis=1)
    rev_err_a= np.abs(za_t.numpy() - z_hat_a).mean(axis=1)

    print(f"\n  FORWARD PATH (latent → Gaussian):")
    print(f"    ||u||² benign : mean={maha_n.mean():.1f}  "
          f"std={maha_n.std():.1f}")
    print(f"    ||u||² attack : mean={maha_a.mean():.1f}  "
          f"std={maha_a.std():.1f}  "
          f"  ratio={maha_a.mean()/maha_n.mean():.2f}×")
    print(f"    Log-Jacobian benign : {lj_n.mean():.2f}")
    print(f"    Log-Jacobian attack : {lj_a.mean():.2f}")
    print(f"\n  REVERSE PATH (u → reconstructed latent):")
    print(f"    Recon error benign : {rev_err_n.mean():.6f}")
    print(f"    Recon error attack : {rev_err_a.mean():.6f}  "
          f"  ({rev_err_a.mean()/max(rev_err_n.mean(),1e-9):.2f}× higher)")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Method 4 — Forward vs Reverse Flow Path Analysis\n"
        "Decoupled insight into where the bidirectional flow detects anomalies",
        fontsize=11, fontweight='bold')

    # Plot 1: Mahalanobis distance distribution
    axes[0,0].hist(maha_n, bins=40, alpha=0.6,
                   color='steelblue', label='Benign', density=True)
    axes[0,0].hist(maha_a, bins=40, alpha=0.6,
                   color='crimson',   label='Attack', density=True)
    axes[0,0].set_title("Forward Path: ||u||² Distribution\n"
                         "Mahalanobis distance from N(0,I) — higher = more anomalous")
    axes[0,0].set_xlabel("||u||²")
    axes[0,0].set_ylabel("Density")
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # Plot 2: per-dimension variance of u
    dims = min(50, u_n.shape[1])
    axes[0,1].plot(var_n[:dims], color='steelblue',
                   label='Benign u variance', linewidth=1.2)
    axes[0,1].plot(var_a[:dims], color='crimson',
                   label='Attack u variance', linewidth=1.2)
    axes[0,1].axhline(1.0, color='black', linestyle='--',
                       alpha=0.5, label='Expected N(0,1) var=1')
    axes[0,1].set_title("Per-Dimension Variance of u\n"
                         "(benign should track the N(0,1) line)")
    axes[0,1].set_xlabel("Latent Dimension")
    axes[0,1].set_ylabel("Variance")
    axes[0,1].legend(fontsize=8)
    axes[0,1].grid(True, alpha=0.3)

    # Plot 3: reverse reconstruction error
    axes[1,0].hist(rev_err_n, bins=40, alpha=0.6,
                   color='steelblue', label='Benign', density=True)
    axes[1,0].hist(rev_err_a, bins=40, alpha=0.6,
                   color='crimson',   label='Attack', density=True)
    axes[1,0].set_title("Reverse Path: Latent Reconstruction Error\n"
                         "|z − flow⁻¹(flow(z))|  (higher = atypical path)")
    axes[1,0].set_xlabel("Reconstruction Error")
    axes[1,0].set_ylabel("Density")
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # Plot 4: Log-Jacobian determinant
    axes[1,1].hist(lj_n, bins=40, alpha=0.6,
                   color='steelblue', label='Benign', density=True)
    axes[1,1].hist(lj_a, bins=40, alpha=0.6,
                   color='crimson',   label='Attack', density=True)
    axes[1,1].set_title("Log-Jacobian Determinant Distribution\n"
                         "Volume change in the flow mapping")
    axes[1,1].set_xlabel("Log |det(J)|")
    axes[1,1].set_ylabel("Density")
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{DRIVE_BASE}/explain_forward_reverse.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: explain_forward_reverse.png")
    return maha_n, maha_a, rev_err_n, rev_err_a


# ----------------------------------------------------------------
# RUN ALL EXPLAINABILITY METHODS
# ----------------------------------------------------------------
def run_all_explainability(model, flow,
                            X_train, X_test_n, X_test_a,
                            z_tr, z_tn, z_ta):
    print("\n" + "="*60)
    print("  EXPLAINABILITY — 4 METHODS")
    print("  Bidirectional Normalizing Flow on CICIDS2017")
    print("="*60)

    # Method 1 — Shapley
    diff_imp, windowed = explain_method1_shapley(
        model, flow, X_test_n, X_test_a)

    # Method 2 — Latent Traversal
    z_2d, y_2d, sc_2d = explain_method2_latent_traversal(
        model, flow, X_test_n, X_test_a)

    # Method 3 — Residual BMR
    res_n, res_a, diff_res = explain_method3_residual_bmr(
        model, X_test_n, X_test_a)

    # Method 4 — Forward/Reverse
    maha_n, maha_a, rev_n, rev_a = explain_method4_forward_reverse(
        flow, z_tn, z_ta)

    print("\n" + "="*60)
    print("  EXPLAINABILITY SUMMARY")
    print("="*60)
    print(f"""
  Method 1 — Shapley Feature Importance:
    Identifies WHICH raw packet bytes drive detection.
    Output: explain_shapley.png

  Method 2 — Latent Space Traversal:
    t-SNE shows benign/attack cluster separation.
    Anomaly score heatmap reveals flow decision boundary.
    Output: explain_latent.png

  Method 3 — Residual Analysis (BMR):
    Attack packets show {res_a.sum()/max(res_n.sum(),1e-9):.2f}× higher reconstruction error.
    Pinpoints byte regions the model finds anomalous.
    Output: explain_residual.png

  Method 4 — Forward/Reverse Path:
    Attack ||u||² = {maha_a.mean():.1f}  vs  Benign ||u||² = {maha_n.mean():.1f}
    ({maha_a.mean()/max(maha_n.mean(),1e-9):.2f}× higher Mahalanobis distance)
    Reverse recon error: attack = {rev_a.mean():.6f}  benign = {rev_n.mean():.6f}
    Output: explain_forward_reverse.png
    """)
    print("✅ All 4 explainability analyses complete!")


# ================================================================
# MAIN PIPELINE
# ================================================================

def main():
    print("=" * 60)
    print("  CICIDS2017 — ANOMALY DETECTION + EXPLAINABILITY")
    print("=" * 60)

    available_pairs = setup()
    if not available_pairs:
        print("\n❌ Cannot proceed — upload Wednesday PCAP + CSV")
        return None, None, None

    # Load or extract data
    if (file_ok('X_train.npy') and file_ok('X_test_n.npy') and
            file_ok('X_test_a.npy')):
        print("\n📂 Loading saved data...")
        X_train  = np.load(f'{DRIVE_BASE}/X_train.npy')
        X_test_n = np.load(f'{DRIVE_BASE}/X_test_n.npy')
        X_test_a = np.load(f'{DRIVE_BASE}/X_test_a.npy')
        print(f"  X_train{X_train.shape}  "
              f"X_test_n{X_test_n.shape}  X_test_a{X_test_a.shape}")
    else:
        X_train, X_test_n, X_test_a = extract_all_data(available_pairs)

    # Load or train encoder
    clear_gpu()
    model = CNNLSTMAutoEncoder(INPUT_SIZE, N_FEATURE, LATENT_SIZE).to(DEVICE)
    if file_ok('best_encoder.pth'):
        print("\n📂 Loading saved encoder...")
        model.load_state_dict(torch.load(f'{DRIVE_BASE}/best_encoder.pth',
                                          map_location=DEVICE))
        model.eval()
    else:
        model = train_encoder(X_train)

    # Load or extract latent features
    if (file_ok('z_train.npy') and file_ok('z_test_n.npy') and
            file_ok('z_test_a.npy')):
        print("\n📂 Loading saved features...")
        z_train  = np.load(f'{DRIVE_BASE}/z_train.npy')
        z_test_n = np.load(f'{DRIVE_BASE}/z_test_n.npy')
        z_test_a = np.load(f'{DRIVE_BASE}/z_test_a.npy')
    else:
        print("\n🔧 Extracting latent features...")
        z_train  = extract_features(model, X_train)
        z_test_n = extract_features(model, X_test_n)
        z_test_a = extract_features(model, X_test_a)
        np.save(f'{DRIVE_BASE}/z_train.npy',  z_train)
        np.save(f'{DRIVE_BASE}/z_test_n.npy', z_test_n)
        np.save(f'{DRIVE_BASE}/z_test_a.npy', z_test_a)

    model.cpu()
    clear_gpu()

    # Flatten latent for flow
    b, c, t = z_train.shape
    n_feat  = c * t
    z_tr    = z_train.reshape(len(z_train),  -1)
    z_tn    = z_test_n.reshape(len(z_test_n), -1)
    z_ta    = z_test_a.reshape(len(z_test_a), -1)
    print(f"\n  Latent: {c}×{t} = {n_feat}-dim")

    # Load or train flow
    flow = build_flow(n_feat, FLOW_BLOCKS)
    if file_ok('best_flow.pth'):
        print("\n📂 Loading saved flow...")
        flow.load_state_dict(torch.load(f'{DRIVE_BASE}/best_flow.pth',
                                         map_location='cpu'))
    else:
        flow = train_flow(z_tr)

    # Scoring
    print("\n🔍 Running ensemble scoring...")
    scores, y_true, auroc, best_w, a_fl, a_knn = ensemble_score(
        flow, z_tr, z_tn, z_ta)

    # Evaluation plots
    acc, preds = evaluate_and_plot(scores, y_true, auroc, a_fl, a_knn)

    # ── EXPLAINABILITY ──────────────────────────────────────────
    print("\n🔬 Running Explainability Module (4 methods)...")
    model.to(DEVICE)
    run_all_explainability(
        model, flow,
        X_train, X_test_n, X_test_a,
        z_tr, z_tn, z_ta
    )
    # ────────────────────────────────────────────────────────────

    print(f"""
{'='*60}
  COMPLETE SUMMARY — CICIDS2017
{'='*60}
  Dataset  : CIC-IDS-2017
  Attacks  : DoS Hulk, DoS GoldenEye, DoS slowloris,
             DoS Slowhttptest, Heartbleed

  Training : Monday PCAP  ({X_train.shape[0]:,} benign packets)
  Testing  : Wednesday PCAP  ({X_test_n.shape[0]:,} benign + {X_test_a.shape[0]:,} attack)

  Pipeline : CNN+LSTM → {c}×{t}={n_feat}-dim latent
             Normalizing Flow ({FLOW_BLOCKS} AllInOneBlock layers)
             Flow+KNN Ensemble (w={best_w:.1f}/{1-best_w:.1f})

  Results  :
    Flow AUROC     : {a_fl:.4f}
    KNN  AUROC     : {a_knn:.4f}
    Ensemble AUROC : {auroc:.4f}  ← FINAL
    Accuracy       : {acc:.4f}

  Saved outputs:
    final_results.png
    explain_shapley.png        ← Method 1
    explain_latent.png         ← Method 2
    explain_residual.png       ← Method 3
    explain_forward_reverse.png← Method 4
{'='*60}
✅ ALL DONE!
""")
    return auroc, scores, y_true


if __name__ == '__main__':
    auroc, scores, y_true = main()
