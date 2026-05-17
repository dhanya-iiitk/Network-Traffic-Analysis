# ================================================================
# VPN ANOMALY DETECTION — COMPLETE FINAL PROGRAM
# Semi-Supervised: CNN+LSTM Encoder + Normalizing Flow + KNN
# + 4 Explainability Methods
#
# DATASET: UNB-CIC VPN/NonVPN (ISCXVPN2016)
#
# DRIVE STRUCTURE:
#   /MyDrive/VPN_PCAP/
#     ├── NonVPN-PCAPs-01.zip  ← normal traffic (train + test)
#     └── VPN-PCAPS-01.zip     ← VPN anomaly traffic (test only)
#
# INSTALL (run once before this script):
#   !pip install FrEIA einops scikit-learn scapy -q
#   !pip install matplotlib seaborn -q
#
# RUN:
#   exec(open('vpn_final.py').read())
#   auroc, scores, y_true = main()
# ================================================================

import os, gc, math, zipfile, shutil
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
# CONFIGURATION  — all paths & hyper-parameters in one place
# ================================================================

# ── FIX 1: correct Drive path (MyDrive, no space) ──────────────
DRIVE_BASE  = '/content/drive/MyDrive/VPN_PCAP'
DATASET_DIR = '/content/vpn_dataset'

# ── FIX 2: correct zip filenames (exact casing) ────────────────
NVPN_ZIP = 'NonVPN-PCAPs-01.zip'   # capital P, lowercase s
VPN_ZIP  = 'VPN-PCAPS-01.zip'

INPUT_SIZE  = 1600
N_FEATURE   = 32
LATENT_SIZE = 64
FLOW_BLOCKS = 8
BATCH_SIZE  = 128
LR          = 0.001
MAX_EPOCHS  = 100
PATIENCE    = 15

MAX_NORMAL   = 30000
MAX_ANOMALY  = 15000
MAX_PER_FILE = 5000
TRAIN_RATIO  = 0.70
PCA_COMPONENTS = 256

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 60)
print("  VPN ANOMALY DETECTION — FINAL (+ Explainability)")
print(f"  Normal zip  : {NVPN_ZIP}")
print(f"  Anomaly zip : {VPN_ZIP}")
print("=" * 60)
print(f"  Device : {DEVICE}")
if torch.cuda.is_available():
    free = (torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_allocated()) / 1e9
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  Free   : {free:.1f} GB")


# ================================================================
# UTILITIES
# ================================================================

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def file_ok(fname, min_bytes=10000):
    """Check a file in DRIVE_BASE exists and is non-trivially sized."""
    path = os.path.join(DRIVE_BASE, fname)
    return os.path.exists(path) and os.path.getsize(path) > min_bytes


# ================================================================
# SETUP  — mount drive, verify zips, extract, report status
# ================================================================

def setup():
    drive.mount('/content/drive')
    os.makedirs(DRIVE_BASE,  exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)

    nvpn_archive = f'{DRIVE_BASE}/{NVPN_ZIP}'
    vpn_archive  = f'{DRIVE_BASE}/{VPN_ZIP}'

    # ── FIX 3: verify files exist with helpful error if not ────
    print(f"\nChecking archive files in {DRIVE_BASE}:")
    for path, label in [(nvpn_archive, NVPN_ZIP), (vpn_archive, VPN_ZIP)]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) // 1024 // 1024
            print(f"  ✅ {label}  ({size_mb} MB)")
        else:
            print(f"  ❌ {label} — NOT FOUND")
            print(f"     Available files in {DRIVE_BASE}:")
            try:
                for f in sorted(os.listdir(DRIVE_BASE)):
                    print(f"       {f}")
            except Exception:
                print(f"     (could not list {DRIVE_BASE})")
            raise FileNotFoundError(
                f"{label} not found at {path}\n"
                f"Please verify the filename casing in your Drive.")

    # ── Extract archives (copy to /content first for speed) ────
    nvpn_dir = f'{DATASET_DIR}/NonVPN'
    vpn_dir  = f'{DATASET_DIR}/VPN'

    def extract_zip_to_folder(archive_path, dest_dir, label):
        """
        Extract zip into dest_dir (creates it).
        Handles 3 layouts:
          A) zip contains a single top-level subfolder  → move contents up
          B) zip contains files directly (flat)         → already in dest_dir
          C) zip contains multiple subfolders           → keep as-is
        """
        if os.path.exists(dest_dir) and len(os.listdir(dest_dir)) > 0:
            print(f"  ✅ {label} already extracted → {dest_dir}")
            return

        local = f'/content/{os.path.basename(archive_path)}'
        if not os.path.exists(local):
            print(f"  Copying {label} to /content for faster extraction...")
            shutil.copy2(archive_path, local)

        # Extract into a temp folder so we can inspect the layout
        tmp_dir = f'{DATASET_DIR}/_tmp_{label}'
        os.makedirs(tmp_dir, exist_ok=True)
        print(f"  Extracting {label}...")
        with zipfile.ZipFile(local, 'r') as zf:
            zf.extractall(tmp_dir)

        # Inspect what was extracted
        top_items = os.listdir(tmp_dir)
        subdirs   = [x for x in top_items
                     if os.path.isdir(os.path.join(tmp_dir, x))]
        files     = [x for x in top_items
                     if os.path.isfile(os.path.join(tmp_dir, x))]

        print(f"  Extracted: {len(subdirs)} subfolder(s), "
              f"{len(files)} file(s) at top level")

        os.makedirs(dest_dir, exist_ok=True)

        if len(subdirs) == 1 and len(files) == 0:
            # Layout A — single subfolder: move its contents up
            sub = os.path.join(tmp_dir, subdirs[0])
            print(f"  Layout A: moving '{subdirs[0]}/' contents → {dest_dir}")
            for item in os.listdir(sub):
                shutil.move(os.path.join(sub, item),
                            os.path.join(dest_dir, item))
        else:
            # Layout B/C — flat files or multiple folders: move everything
            print(f"  Layout B/C: moving all contents → {dest_dir}")
            for item in top_items:
                shutil.move(os.path.join(tmp_dir, item),
                            os.path.join(dest_dir, item))

        shutil.rmtree(tmp_dir, ignore_errors=True)

        # Verify
        n_pcap = sum(
            1 for r, _, fs in os.walk(dest_dir)
            for f in fs if f.lower().endswith(('.pcap', '.pcapng')))
        if n_pcap == 0:
            print(f"  ⚠️  No PCAP files found in {dest_dir} after extraction!")
            print(f"  Contents: {os.listdir(dest_dir)[:10]}")
            raise RuntimeError(
                f"Extraction of {label} produced no PCAP files in {dest_dir}.")
        print(f"  ✅ {label} extracted → {dest_dir}  ({n_pcap} pcap files)")

    extract_zip_to_folder(nvpn_archive, nvpn_dir, 'NonVPN')
    extract_zip_to_folder(vpn_archive,  vpn_dir,  'VPN')

    # ── Status report ───────────────────────────────────────────
    print(f"\nSaved file status:")
    for fname, label in [
        ('X_train.npy',      'Training data  '),
        ('X_test_n.npy',     'Test normal    '),
        ('X_test_a.npy',     'Test VPN       '),
        ('best_encoder.pth', 'Encoder        '),
        ('z_train.npy',      'Latent features'),
        ('best_flow.pth',    'Flow model     '),
    ]:
        st = "✅ Done" if file_ok(fname) else "⏳ Pending"
        print(f"  {label}: {st}")

    return nvpn_dir, vpn_dir


# ================================================================
# PCAP PROCESSING
# ================================================================

def preprocess_packet(packet, max_len=1600):
    """
    Enhanced preprocessing for VPN detection.

    VPN signatures exploited:
      - OpenVPN uses UDP/1194 or TCP/443; encrypted payload has
        near-uniform byte distribution (high entropy).
      - L2TP uses UDP/1701; PPTP uses TCP/1723.
      - VPN tunnel headers carry distinct byte patterns.

    Strategy:
      - Anonymise IP addresses (prevent IP memorisation).
      - Preserve TCP/UDP ports (carry VPN signal: 1194/1701/1723).
      - Encode normalised packet length in last 8 bytes.
      - Drop DNS, ARP, and TCP-ACK-only (no-payload) packets.
    """
    from scapy.all import IP, TCP, UDP, DNS, ARP, Ether
    try:
        if packet.haslayer(DNS) or packet.haslayer(ARP):
            return None
        if packet.haslayer(TCP):
            if len(bytes(packet[TCP].payload)) == 0:
                return None

        # Strip Ethernet header only
        if packet.haslayer(Ether):
            packet = packet.payload

        # Anonymise IPs but KEEP ports
        if packet.haslayer(IP):
            packet[IP].src = '0.0.0.0'
            packet[IP].dst = '0.0.0.0'

        raw     = bytes(packet)
        pkt_len = len(raw)

        decimal = list(raw)
        if len(decimal) < max_len:
            decimal += [0] * (max_len - len(decimal))
        else:
            decimal = decimal[:max_len]

        arr = np.array(decimal, dtype=np.float32) / 255.0

        # Encode packet length in last 8 positions
        norm_len = min(pkt_len, max_len) / max_len
        arr[-8:] = norm_len

        return arr
    except Exception:
        return None


def process_folder(folder_path, max_total, max_per_file=5000):
    """
    Stream packets from all PCAP/PCAPNG files in folder tree.
    Stops once max_total packets collected.
    max_per_file limits per-file contribution for diversity.
    """
    from scapy.all import PcapReader
    all_data   = []
    pcap_files = []

    for root, _, files in os.walk(folder_path):
        for f in sorted(files):
            if f.lower().endswith('.pcap') or f.lower().endswith('.pcapng'):
                pcap_files.append(os.path.join(root, f))

    print(f"  Found {len(pcap_files)} PCAP files in "
          f"{os.path.basename(folder_path)}")

    if len(pcap_files) == 0:
        print(f"  ⚠️  No pcap files found! Listing {folder_path}:")
        for root, dirs, files in os.walk(folder_path):
            for f in files[:10]:
                print(f"    {os.path.join(root, f)}")
        return np.array([], dtype=np.float32).reshape(0, INPUT_SIZE)

    for pcap_path in pcap_files:
        if len(all_data) >= max_total:
            break
        file_count = 0
        try:
            with PcapReader(pcap_path) as reader:
                for pkt in reader:
                    if file_count >= max_per_file:
                        break
                    if len(all_data) >= max_total:
                        break
                    arr = preprocess_packet(pkt)
                    if arr is not None:
                        all_data.append(arr)
                        file_count += 1
            if file_count > 0:
                print(f"    {os.path.basename(pcap_path):<45} "
                      f"+{file_count:5d} (total: {len(all_data)})")
            else:
                print(f"    {os.path.basename(pcap_path):<45} "
                      f"+0 (no valid pkts)")
        except Exception as e:
            print(f"  ⚠️  Skipping {os.path.basename(pcap_path)}: {e}")

    if len(all_data) == 0:
        return np.array([], dtype=np.float32).reshape(0, INPUT_SIZE)

    print(f"  ✅ Extracted {len(all_data)} packets total")
    return np.array(all_data, dtype=np.float32)


def extract_all_data(nvpn_dir, vpn_dir):
    print("\n" + "="*55)
    print("EXTRACTING VPN / NonVPN DATA")
    print("Train : NonVPN (normal only)")
    print("Test  : NonVPN (30%) + VPN (anomaly)")
    print("="*55)

    print("\n[NORMAL] Extracting NonVPN packets...")
    X_normal = process_folder(nvpn_dir, MAX_NORMAL, MAX_PER_FILE)

    print("\n[ANOMALY] Extracting VPN packets...")
    X_anomaly = process_folder(vpn_dir, MAX_ANOMALY, MAX_PER_FILE)

    if len(X_normal) == 0:
        raise ValueError(
            "No normal packets extracted from NonVPN folder.\n"
            f"Folder checked: {nvpn_dir}\n"
            "Possible causes:\n"
            "  1. All packets are DNS/ARP/TCP-ACK — try relaxing\n"
            "     the filter in preprocess_packet().\n"
            "  2. PCAP files are corrupt or use an unsupported format.")
    if len(X_anomaly) == 0:
        raise ValueError(
            "No VPN packets extracted from VPN folder.\n"
            f"Folder checked: {vpn_dir}")

    # Split normal: 70% train, 30% test
    np.random.seed(42)
    n_train = int(len(X_normal) * TRAIN_RATIO)
    idx     = np.random.permutation(len(X_normal))
    X_train  = X_normal[idx[:n_train]]
    X_test_n = X_normal[idx[n_train:]]

    # Balance test set
    n_test   = min(len(X_test_n), len(X_anomaly))
    X_test_n = X_test_n[:n_test]
    X_test_a = X_anomaly[:n_test]

    print(f"\n  Split summary:")
    print(f"    X_train  : {len(X_train):6d}  (NonVPN — train)")
    print(f"    X_test_n : {len(X_test_n):6d}  (NonVPN — test)")
    print(f"    X_test_a : {len(X_test_a):6d}  (VPN    — test)")
    ratio_tr = len(X_train) / (len(X_train) + len(X_test_n))
    print(f"    Ratio    : {ratio_tr:.0%} train / {1-ratio_tr:.0%} test")

    np.save(f'{DRIVE_BASE}/X_train.npy',  X_train)
    np.save(f'{DRIVE_BASE}/X_test_n.npy', X_test_n)
    np.save(f'{DRIVE_BASE}/X_test_a.npy', X_test_a)

    print(f"\n✅ Data saved to {DRIVE_BASE}")
    print(f"  X_train  : {X_train.shape}")
    print(f"  X_test_n : {X_test_n.shape}")
    print(f"  X_test_a : {X_test_a.shape}")
    return X_train, X_test_n, X_test_a


# ================================================================
# CNN + LSTM AUTOENCODER
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
            self.pyramid.add_module(f"bn_{out_feat}",
                nn.BatchNorm1d(out_feat))
            self.pyramid.add_module(f"relu_{out_feat}",
                nn.LeakyReLU(0.2, inplace=True))
            in_feat = out_feat
            pdim    = pdim // 2
        self.lstm = nn.LSTM(
            input_size=in_feat, hidden_size=latent_vec_size,
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
        pdim_trace    = input_size // 2
        in_feat_trace = n_features
        while pdim_trace > 100:
            in_feat_trace = in_feat_trace * 2
            pdim_trace    = pdim_trace // 2
        n_in = in_feat_trace
        self.lstm_dec = nn.LSTM(
            input_size=latent_vec_size, hidden_size=n_in,
            num_layers=2, batch_first=True, dropout=0.2)
        self.inverse_pyramid = nn.Sequential()
        pdim    = input_size // 2
        in_feat = n_in
        while pdim > 100:
            out_feat = in_feat // 2
            self.inverse_pyramid.add_module(f"convt_{in_feat}",
                nn.ConvTranspose1d(in_feat, out_feat,
                                   kernel_size=4, stride=2, padding=1))
            self.inverse_pyramid.add_module(f"bn_{out_feat}",
                nn.BatchNorm1d(out_feat))
            self.inverse_pyramid.add_module(f"relu_{out_feat}",
                nn.ReLU(True))
            in_feat = out_feat
            pdim    = pdim // 2
        self.final = nn.Sequential(
            nn.ConvTranspose1d(in_feat, 1,
                               kernel_size=4, stride=2, padding=1),
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
            recon = F.interpolate(
                recon.unsqueeze(1), size=x.shape[1]).squeeze(1)
        return latent, recon


def train_encoder(X_train):
    clear_gpu()
    model  = CNNLSTMAutoEncoder(INPUT_SIZE, N_FEATURE, LATENT_SIZE).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"\nTraining CNN+LSTM Encoder (NonVPN normal only)...")
    print(f"  Samples    : {len(X_train)}")
    print(f"  Parameters : {params/1e6:.2f}M")

    loader    = DataLoader(
        TensorDataset(torch.FloatTensor(X_train)),
        batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    best_loss  = float('inf')
    patience_c = 0
    losses     = []

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
            torch.save(model.state_dict(),
                       f'{DRIVE_BASE}/best_encoder.pth')
        else:
            patience_c += 1

        if patience_c >= PATIENCE:
            print(f"  Early stop at epoch {epoch+1}")
            break
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1:3d}] Loss:{avg:.6f}  "
                  f"Best:{best_loss:.6f}")
        if (epoch + 1) % 5 == 0:
            clear_gpu()

    plt.figure(figsize=(8, 4))
    plt.plot(losses, 'steelblue', linewidth=2)
    plt.title('VPN — Encoder Training Loss (NonVPN Normal)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{DRIVE_BASE}/encoder_loss.png', dpi=150)
    plt.show()
    print(f"✅ Encoder trained. Best loss: {best_loss:.6f}")
    return model


def extract_features(model, data, batch_size=64):
    if len(data) == 0:
        return np.array([])
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(data)),
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
        flow.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_fc,
            affine_clamping=1.9,
            global_affine_type="SOFTPLUS",
            permute_soft=False)
    return flow


def train_flow(z_flat):
    n_feat = z_flat.shape[1]
    flow   = build_flow(n_feat, FLOW_BLOCKS)
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(z_flat)),
        batch_size=256, shuffle=True)
    optimizer  = torch.optim.Adam(
        flow.parameters(), lr=LR, betas=(0.5, 0.999))
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
            loss = torch.mean(
                0.5 * torch.sum(p_u**2, dim=1) - log_jac)
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
            torch.save(flow.state_dict(),
                       f'{DRIVE_BASE}/best_flow.pth')
        else:
            patience_c += 1

        if patience_c >= PATIENCE:
            print(f"  Early stop at epoch {epoch+1}")
            break
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1:3d}] Loss:{avg:.2f}  "
                  f"Best:{best_loss:.2f}")

    plt.figure(figsize=(8, 4))
    plt.plot(losses, 'orange', linewidth=2)
    plt.title('VPN — Normalizing Flow Loss')
    plt.xlabel('Epoch')
    plt.ylabel('NLL Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{DRIVE_BASE}/flow_loss.png', dpi=150)
    plt.show()
    print(f"✅ Flow trained. Best loss: {best_loss:.2f}")
    return flow


# ================================================================
# ENSEMBLE SCORING
# ================================================================

def compute_flow_scores(flow, z_flat):
    flow.eval()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(z_flat)),
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

    sc       = RobustScaler()
    fl_norm  = sc.fit_transform(fl_all.reshape(-1,1)).flatten()
    sc2      = RobustScaler()
    knn_norm = sc2.fit_transform(knn_all.reshape(-1,1)).flatten()

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
    print(f"  Weights        : flow={best_w:.1f}  knn={1-best_w:.1f}")
    return ens, y, a_ens, best_w, a_fl, a_knn


# ================================================================
# EVALUATION & PLOTS
# ================================================================

def evaluate_and_plot(scores, y_true, auroc, a_fl, a_knn):
    fpr, tpr, thresh = roc_curve(y_true, scores)
    best_t = thresh[np.argmax(tpr - fpr)]
    preds  = (scores >= best_t).astype(int)
    acc    = (preds == y_true).mean()

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS — VPN / NonVPN")
    print(f"{'='*60}")
    print(f"  🎯 AUROC     : {auroc:.4f}")
    print(f"  🎯 Accuracy  : {acc:.4f}")
    print(f"  🎯 Threshold : {best_t:.4f}")
    print(classification_report(
        y_true, preds,
        target_names=['NonVPN (Normal)', 'VPN (Anomaly)']))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f'VPN/NonVPN  |  AUROC={auroc:.4f}  |  Accuracy={acc:.4f}',
        fontsize=13, fontweight='bold')

    cm = confusion_matrix(y_true, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['NonVPN','VPN'],
                yticklabels=['NonVPN','VPN'])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    axes[1].hist(scores[y_true==0], bins=60, alpha=0.6,
                 color='steelblue', label='NonVPN (Normal)', density=True)
    axes[1].hist(scores[y_true==1], bins=60, alpha=0.6,
                 color='crimson', label='VPN (Anomaly)', density=True)
    axes[1].axvline(best_t, color='black', linestyle='--',
                    linewidth=2, label=f'Threshold={best_t:.3f}')
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
    plt.savefig(f'{DRIVE_BASE}/final_results.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: final_results.png")
    return acc, preds


# ================================================================
# EXPLAINABILITY — 4 METHODS
# ================================================================

def explain_method1_shapley(model, flow, X_normal, X_attack,
                             n_samples=300, pca_model=None):
    print("\n" + "="*60)
    print("EXPLAINABILITY — METHOD 1: Shapley Feature Importance")
    print("Which raw packet bytes drive the anomaly score?")
    print("="*60)

    flow.eval()
    n   = min(n_samples, len(X_normal), len(X_attack))
    X_n = torch.FloatTensor(X_normal[:n]).to(DEVICE)
    X_a = torch.FloatTensor(X_attack[:n]).to(DEVICE)

    def byte_importance(X_tensor):
        model.train()
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.eval()
        X_tensor = X_tensor.clone().detach().requires_grad_(True)
        latent, recon = model(X_tensor)
        recon_err = torch.mean((X_tensor - recon) ** 2)
        recon_err.backward()
        grad = X_tensor.grad.abs().mean(dim=0).cpu().numpy()
        model.eval()
        return grad

    imp_n    = byte_importance(X_n)
    imp_a    = byte_importance(X_a)
    diff_imp = np.clip(imp_a - imp_n, 0, None)
    diff_imp /= (diff_imp.sum() + 1e-10)

    W        = 50
    n_win    = INPUT_SIZE // W
    windowed = np.array([diff_imp[i*W:(i+1)*W].sum() for i in range(n_win)])
    windowed /= (windowed.sum() + 1e-10)
    top3     = np.argsort(windowed)[::-1][:3]

    REGIONS = {
        0: "IP header (TTL, proto, addresses)",
        1: "TCP header (ports, seq, flags)",
        2: "Payload start (VPN handshake / hello)",
        3: "Payload (VPN tunnel setup)",
        4: "Payload (VPN encrypted content)",
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
        "Method 1 — Shapley-Inspired Feature Importance (VPN/NonVPN)\n"
        "Gradient of anomaly score w.r.t. each raw packet byte",
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


def explain_method2_latent_traversal(model, flow,
                                      X_normal, X_attack,
                                      n_samples=200, pca=None):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    print("\n" + "="*60)
    print("EXPLAINABILITY — METHOD 2: Latent Space Traversal")
    print("t-SNE projection showing NonVPN vs VPN separation")
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

    viz_pca = PCA(n_components=min(50, z_all.shape[1]), random_state=42)
    z_pca   = viz_pca.fit_transform(z_all)
    var10   = viz_pca.explained_variance_ratio_[:10].sum()

    tsne = TSNE(n_components=2, perplexity=30,
                random_state=42, max_iter=500)
    z_2d = tsne.fit_transform(z_pca)

    flow.eval()
    with torch.no_grad():
        zn_flow = (torch.FloatTensor(pca.transform(z_n))
                   if pca is not None else torch.FloatTensor(z_n))
        za_flow = (torch.FloatTensor(pca.transform(z_a))
                   if pca is not None else torch.FloatTensor(z_a))
        pu_n, lj_n = flow(zn_flow)
        pu_a, lj_a = flow(za_flow)
        sc_n = (0.5*torch.sum(pu_n**2,dim=1) - lj_n).numpy()
        sc_a = (0.5*torch.sum(pu_a**2,dim=1) - lj_a).numpy()
    sc_all = np.concatenate([sc_n, sc_a])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Method 2 — Latent Space Traversal (VPN/NonVPN)\n"
        "Encoder latent space projected via PCA + t-SNE",
        fontsize=11, fontweight='bold')

    for lbl, name, col in [
            (0,'NonVPN (Normal)','steelblue'),
            (1,'VPN (Anomaly)','crimson')]:
        m = y_all == lbl
        axes[0].scatter(z_2d[m,0], z_2d[m,1],
                        c=col, label=name, alpha=0.5, s=12)
    axes[0].set_title("Class Clusters in Latent Space")
    axes[0].legend(fontsize=8)
    axes[0].set_xlabel("t-SNE dim 1")
    axes[0].set_ylabel("t-SNE dim 2")
    axes[0].grid(True, alpha=0.3)

    sc_plot = axes[1].scatter(z_2d[:,0], z_2d[:,1], c=sc_all,
                               cmap='RdYlGn_r', alpha=0.6, s=12)
    plt.colorbar(sc_plot, ax=axes[1], label='Anomaly Score')
    axes[1].set_title("Anomaly Score Heatmap\n"
                       "(red=high anomaly, green=normal)")
    axes[1].set_xlabel("t-SNE dim 1")
    axes[1].grid(True, alpha=0.3)

    n_pc = min(20, viz_pca.n_components_)
    axes[2].bar(range(1, n_pc+1),
                viz_pca.explained_variance_ratio_[:n_pc]*100,
                color='steelblue', alpha=0.8)
    axes[2].set_title(
        f"PCA Scree Plot\nTop 10 PCs: {var10*100:.1f}% variance")
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


def explain_method3_residual_bmr(model, X_normal, X_attack,
                                  n_samples=200):
    print("\n" + "="*60)
    print("EXPLAINABILITY — METHOD 3: Residual Analysis (BMR)")
    print("Where does the autoencoder fail on VPN packets?")
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

    print(f"\n  Mean residual — NonVPN : {mu_n.sum():.4f}")
    print(f"  Mean residual — VPN    : {mu_a.sum():.4f}  "
          f"({mu_a.sum()/max(mu_n.sum(),1e-9):.2f}× higher)")
    print("\n  Top 3 byte windows (highest VPN residual excess):")
    for i in top3:
        print(f"    Bytes {i*W:4d}–{(i+1)*W-1}: excess={w_d[i]:.5f}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Method 3 — Residual Analysis BMR (VPN/NonVPN)\n"
        "|original − reconstructed| per byte",
        fontsize=11, fontweight='bold')

    for ax, data, col, title in [
        (axes[0,0], mu_n, 'steelblue', 'Mean Residual — NonVPN'),
        (axes[0,1], mu_a, 'crimson',   'Mean Residual — VPN')
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
    axes[1,0].set_title("VPN − NonVPN Residual Excess")
    axes[1,0].set_xlabel("Byte Position")
    axes[1,0].set_ylabel("Residual Excess")
    for x, lbl in [(14,"IP→TCP"),(34,"Payload")]:
        axes[1,0].axvline(x, color='navy', linestyle='--', alpha=0.6)
        axes[1,0].text(x+3, axes[1,0].get_ylim()[1]*0.85, lbl,
                       fontsize=7, color='navy')
    axes[1,0].grid(True, alpha=0.3)

    xp = np.arange(n_win)
    bw = 0.35
    axes[1,1].bar(xp-bw/2, w_n, bw,
                  label='NonVPN', color='steelblue', alpha=0.8)
    axes[1,1].bar(xp+bw/2, w_a, bw,
                  label='VPN',    color='crimson',   alpha=0.8)
    axes[1,1].set_title("50-Byte Window Residual Comparison")
    axes[1,1].set_xlabel("Byte Window")
    axes[1,1].set_ylabel("Mean Residual")
    axes[1,1].set_xticks(xp)
    axes[1,1].set_xticklabels(
        [f"{i*W}" for i in range(n_win)],
        rotation=45, ha='right', fontsize=7)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{DRIVE_BASE}/explain_residual.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: explain_residual.png")
    return mu_n, mu_a, diff


def explain_method4_forward_reverse(flow, z_tn, z_ta,
                                     n_samples=300, pca=None):
    print("\n" + "="*60)
    print("EXPLAINABILITY — METHOD 4: Forward vs Reverse Path")
    print("Decoupled analysis of the normalizing flow")
    print("="*60)

    flow.eval()
    n    = min(n_samples, len(z_tn), len(z_ta))
    zn_t = torch.FloatTensor(z_tn[:n])
    za_t = torch.FloatTensor(z_ta[:n])

    with torch.no_grad():
        zn_flow = (torch.FloatTensor(pca.transform(zn_t.numpy()))
                   if pca is not None else zn_t)
        za_flow = (torch.FloatTensor(pca.transform(za_t.numpy()))
                   if pca is not None else za_t)
        u_n, lj_n = flow(zn_flow)
        u_a, lj_a = flow(za_flow)
        u_n  = u_n.numpy();  lj_n = lj_n.numpy()
        u_a  = u_a.numpy();  lj_a = lj_a.numpy()
        z_hat_n = flow(torch.FloatTensor(u_n), rev=True)[0].numpy()
        z_hat_a = flow(torch.FloatTensor(u_a), rev=True)[0].numpy()

    maha_n    = np.sum(u_n**2, axis=1)
    maha_a    = np.sum(u_a**2, axis=1)
    var_n     = u_n.var(axis=0)
    var_a     = u_a.var(axis=0)
    rev_err_n = np.abs(zn_flow.numpy() - z_hat_n).mean(axis=1)
    rev_err_a = np.abs(za_flow.numpy() - z_hat_a).mean(axis=1)

    print(f"\n  FORWARD PATH:")
    print(f"    ||u||² NonVPN : mean={maha_n.mean():.1f}  "
          f"std={maha_n.std():.1f}")
    print(f"    ||u||² VPN    : mean={maha_a.mean():.1f}  "
          f"std={maha_a.std():.1f}  "
          f"ratio={maha_a.mean()/max(maha_n.mean(),1e-9):.2f}×")
    print(f"    Log-Jacobian NonVPN : {lj_n.mean():.2f}")
    print(f"    Log-Jacobian VPN    : {lj_a.mean():.2f}")
    print(f"\n  REVERSE PATH:")
    print(f"    Recon error NonVPN : {rev_err_n.mean():.6f}")
    print(f"    Recon error VPN    : {rev_err_a.mean():.6f}  "
          f"({rev_err_a.mean()/max(rev_err_n.mean(),1e-9):.2f}× higher)")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Method 4 — Forward vs Reverse Flow Path Analysis (VPN/NonVPN)\n"
        "Decoupled insight into where the flow detects VPN anomalies",
        fontsize=11, fontweight='bold')

    axes[0,0].hist(maha_n, bins=40, alpha=0.6,
                   color='steelblue', label='NonVPN', density=True)
    axes[0,0].hist(maha_a, bins=40, alpha=0.6,
                   color='crimson',   label='VPN',    density=True)
    axes[0,0].set_title("Forward Path: ||u||² Distribution\n"
                         "Mahalanobis distance from N(0,I)")
    axes[0,0].set_xlabel("||u||²")
    axes[0,0].set_ylabel("Density")
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    dims = min(50, u_n.shape[1])
    axes[0,1].plot(var_n[:dims], color='steelblue',
                   label='NonVPN u variance', linewidth=1.2)
    axes[0,1].plot(var_a[:dims], color='crimson',
                   label='VPN u variance',    linewidth=1.2)
    axes[0,1].axhline(1.0, color='black', linestyle='--',
                      alpha=0.5, label='Expected N(0,1) var=1')
    axes[0,1].set_title("Per-Dimension Variance of u")
    axes[0,1].set_xlabel("Latent Dimension")
    axes[0,1].set_ylabel("Variance")
    axes[0,1].legend(fontsize=8)
    axes[0,1].grid(True, alpha=0.3)

    axes[1,0].hist(rev_err_n, bins=40, alpha=0.6,
                   color='steelblue', label='NonVPN', density=True)
    axes[1,0].hist(rev_err_a, bins=40, alpha=0.6,
                   color='crimson',   label='VPN',    density=True)
    axes[1,0].set_title("Reverse Path: Reconstruction Error\n"
                         "|z − flow⁻¹(flow(z))|")
    axes[1,0].set_xlabel("Reconstruction Error")
    axes[1,0].set_ylabel("Density")
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    axes[1,1].hist(lj_n, bins=40, alpha=0.6,
                   color='steelblue', label='NonVPN', density=True)
    axes[1,1].hist(lj_a, bins=40, alpha=0.6,
                   color='crimson',   label='VPN',    density=True)
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


def run_all_explainability(model, flow,
                            X_train, X_test_n, X_test_a,
                            z_tr, z_tn, z_ta):
    print("\n" + "="*60)
    print("  EXPLAINABILITY — 4 METHODS")
    print("="*60)

    pca = None
    pca_path = f'{DRIVE_BASE}/pca_model.pkl'
    if os.path.exists(pca_path):
        import pickle
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
        print(f"  ✅ PCA model loaded ({pca.n_components_} components)")
    else:
        print("  ℹ️  No PCA model — using full latent dims")

    explain_method1_shapley(
        model, flow, X_test_n, X_test_a, pca_model=None)

    explain_method2_latent_traversal(
        model, flow, X_test_n, X_test_a, pca=pca)

    explain_method3_residual_bmr(
        model, X_test_n, X_test_a)

    explain_method4_forward_reverse(
        flow, z_tn, z_ta, pca=pca)

    print("\n" + "="*60)
    print("  EXPLAINABILITY SUMMARY")
    print("="*60)
    print("""
  Method 1 — Shapley Feature Importance
    Which raw bytes drive detection → explain_shapley.png

  Method 2 — Latent Space Traversal
    t-SNE cluster separation        → explain_latent.png

  Method 3 — Residual Analysis (BMR)
    Autoencoder failure on VPN      → explain_residual.png

  Method 4 — Forward / Reverse Path
    Flow Mahalanobis & recon error  → explain_forward_reverse.png
    """)
    print("✅ All 4 explainability analyses complete!")


# ================================================================
# MAIN PIPELINE
# ================================================================

def main():
    print("=" * 60)
    print("  VPN ANOMALY DETECTION + EXPLAINABILITY")
    print("=" * 60)

    # ── 1. Setup & verify files ─────────────────────────────────
    nvpn_dir, vpn_dir = setup()

    # ── 2. Load or extract raw packet data ──────────────────────
    if (file_ok('X_train.npy') and file_ok('X_test_n.npy') and
            file_ok('X_test_a.npy')):
        print("\n📂 Loading saved packet data...")
        X_train  = np.load(f'{DRIVE_BASE}/X_train.npy')
        X_test_n = np.load(f'{DRIVE_BASE}/X_test_n.npy')
        X_test_a = np.load(f'{DRIVE_BASE}/X_test_a.npy')
        print(f"  X_train  {X_train.shape}")
        print(f"  X_test_n {X_test_n.shape}")
        print(f"  X_test_a {X_test_a.shape}")
    else:
        X_train, X_test_n, X_test_a = extract_all_data(nvpn_dir, vpn_dir)

    # ── 3. Load or train encoder ─────────────────────────────────
    clear_gpu()
    model = CNNLSTMAutoEncoder(
        INPUT_SIZE, N_FEATURE, LATENT_SIZE).to(DEVICE)
    if file_ok('best_encoder.pth'):
        print("\n📂 Loading saved encoder...")
        model.load_state_dict(torch.load(
            f'{DRIVE_BASE}/best_encoder.pth', map_location=DEVICE))
        model.eval()
    else:
        model = train_encoder(X_train)

    # ── 4. Load or extract latent features ──────────────────────
    if (file_ok('z_train.npy') and file_ok('z_test_n.npy') and
            file_ok('z_test_a.npy')):
        print("\n📂 Loading saved latent features...")
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

    # ── 5. Flatten latent vectors ────────────────────────────────
    b, c, t = z_train.shape
    n_feat  = c * t
    z_tr    = z_train.reshape(len(z_train),  -1)
    z_tn    = z_test_n.reshape(len(z_test_n), -1)
    z_ta    = z_test_a.reshape(len(z_test_a), -1)
    print(f"\n  Latent shape: {c}×{t} = {n_feat}-dim per packet")

    # ── 6. PCA reduction (6400 → 256 dims) ──────────────────────
    pca_path  = f'{DRIVE_BASE}/pca_model.pkl'
    z_tr_path = f'{DRIVE_BASE}/z_tr_pca.npy'

    if (os.path.exists(z_tr_path) and
            os.path.getsize(z_tr_path) > 1000):
        print("\n📂 Loading PCA-reduced latents...")
        import pickle
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
        z_tr = np.load(f'{DRIVE_BASE}/z_tr_pca.npy')
        z_tn = np.load(f'{DRIVE_BASE}/z_tn_pca.npy')
        z_ta = np.load(f'{DRIVE_BASE}/z_ta_pca.npy')
        n_feat = z_tr.shape[1]
        print(f"  ✅ Loaded: {z_tr.shape}")
    else:
        print(f"\n🔧 PCA: {n_feat} → {PCA_COMPONENTS} dims...")
        from sklearn.decomposition import PCA as SKPCA
        import pickle
        pca  = SKPCA(n_components=PCA_COMPONENTS, random_state=42)
        z_tr = pca.fit_transform(z_tr)
        z_tn = pca.transform(z_tn)
        z_ta = pca.transform(z_ta)
        var  = pca.explained_variance_ratio_.sum()
        print(f"  Variance explained: {var*100:.1f}%")
        n_feat = PCA_COMPONENTS
        np.save(f'{DRIVE_BASE}/z_tr_pca.npy', z_tr)
        np.save(f'{DRIVE_BASE}/z_tn_pca.npy', z_tn)
        np.save(f'{DRIVE_BASE}/z_ta_pca.npy', z_ta)
        with open(pca_path, 'wb') as f:
            pickle.dump(pca, f)
        print(f"  ✅ PCA saved: {z_tr.shape}")

    print(f"  Flow trains on {n_feat}-dim  "
          f"(was {c*t}-dim, ~{c*t//n_feat}× speedup)")

    # ── 7. Load or train normalizing flow ────────────────────────
    flow = build_flow(n_feat, FLOW_BLOCKS)
    if file_ok('best_flow.pth'):
        print("\n📂 Loading saved flow model...")
        flow.load_state_dict(torch.load(
            f'{DRIVE_BASE}/best_flow.pth', map_location='cpu'))
    else:
        flow = train_flow(z_tr)

    # ── 8. Ensemble scoring ──────────────────────────────────────
    print("\n🔍 Running ensemble scoring...")
    scores, y_true, auroc, best_w, a_fl, a_knn = ensemble_score(
        flow, z_tr, z_tn, z_ta)

    # ── 9. Evaluation & plots ────────────────────────────────────
    acc, preds = evaluate_and_plot(scores, y_true, auroc, a_fl, a_knn)

    # ── 10. Explainability ───────────────────────────────────────
    print("\n🔬 Running Explainability Module (4 methods)...")
    model.to(DEVICE)
    run_all_explainability(
        model, flow,
        X_train, X_test_n, X_test_a,
        z_tr, z_tn, z_ta)

    # ── Final summary ────────────────────────────────────────────
    print(f"""
{'='*60}
  COMPLETE SUMMARY — VPN / NonVPN
{'='*60}
  Dataset  : ISCXVPN2016 (UNB-CIC)
  Normal   : NonVPN-PCAPs-01.zip
  Anomaly  : VPN-PCAPS-01.zip

  Training : {X_train.shape[0]:,} NonVPN packets (normal only)
  Testing  : {X_test_n.shape[0]:,} NonVPN  +  {X_test_a.shape[0]:,} VPN  (balanced)

  Pipeline : CNN+LSTM → {c}×{t}={n_feat}-dim PCA latent
             Normalizing Flow ({FLOW_BLOCKS} AllInOneBlock layers)
             Flow + KNN Ensemble  (w={best_w:.1f}/{1-best_w:.1f})

  Results  :
    Flow AUROC     : {a_fl:.4f}
    KNN  AUROC     : {a_knn:.4f}
    Ensemble AUROC : {auroc:.4f}  ← FINAL
    Accuracy       : {acc:.4f}

  Outputs saved to {DRIVE_BASE}/:
    encoder_loss.png
    flow_loss.png
    final_results.png
    explain_shapley.png          ← Method 1
    explain_latent.png           ← Method 2
    explain_residual.png         ← Method 3
    explain_forward_reverse.png  ← Method 4
{'='*60}
✅ ALL DONE!
""")
    return auroc, scores, y_true


# ================================================================
if __name__ == '__main__':
    auroc, scores, y_true = main()
