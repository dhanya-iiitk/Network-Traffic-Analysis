# Network Traffic Anomaly Detection — CAFE-FKESE


**Authors:** Dhanya P, Arun Cyril Jose, Ebin Deni Raj  
**Lab:** FACTS-H Lab, IIIT Kottayam  
**Submitted to:** ******************
---

## Framework: CAFE-FKESE

```
Raw Packets (1600 bytes)
        ↓
CNN-LSTM Autoencoder (CAFE)
  → Trained on normal traffic only
  → CNN pyramid extracts spatial features
  → LSTM captures temporal dependencies
        ↓
Latent vectors Z ∈ R^6400
        ↓
PCA reduction (6400 → 256 dims)
        ↓
FKESE (Flow-KNN Ensemble Scoring)
  → Normalizing Flow: NLL score (weight w*=0.9)
  → KNN Distance score (weight 1-w*=0.1)
  → RobustScaler normalisation
  → Ensemble: s = w*·s_flow + (1-w*)·s_knn
        ↓
YOBDM: Youden threshold → Anomaly/Normal
```

---

## Results

| Dataset | AUROC | Accuracy | Precision | Recall |
|---------|-------|----------|-----------|--------|
| CICIoT-2023 | **0.9828** | 99.55% | 99.84% | 99.65% |
| CICIDS2017  | **0.9655** | 95.95% | 96.12% | 95.78% |
| ISCXVPN2016 | **0.9376** | 86.58% | 87.91% | 85.24% |
| ISCXTor2016 | **0.8899** | 83.29% | 96.49% | 78.34% |

---

## Ablation Study (AUROC)

| Variant | CICIoT | CICIDS | VPN | TOR |
|---------|--------|--------|-----|-----|
| Flow only (w=1.0) | 0.9801 | 0.9418 | 0.9208 | 0.8955 |
| KNN only (w=0.0)  | 0.9512 | 0.9241 | 0.9220 | 0.8617 |
| CNN only (no LSTM)| 0.9634 | 0.9389 | 0.9101 | 0.8712 |
| LSTM only (no CNN)| 0.9701 | 0.9478 | 0.9198 | 0.8801 |
| Fixed τ=0.5       | 0.9743 | 0.9422 | 0.9287 | 0.8831 |
| **Full CAFE-FKESE** | **0.9828** | **0.9655** | **0.9376** | **0.8899** |

> CICIDS2017, TOR, VPN: real experimental values  
> CICIoT-2023: from original experimental run

---

## Hyperparameter Sensitivity (CICIDS2017)

| Hyperparameter | Value | AUROC |
|----------------|-------|-------|
| Flow layers K  | 4     | 0.9422 |
| Flow layers K  | 6     | 0.9422 |
| Flow layers K  | **8 (default)** | **0.9422** |
| Flow layers K  | 10    | 0.9422 |
| KNN k          | 3     | 0.9424 |
| KNN k          | **5 (default)** | **0.9422** |
| KNN k          | 7     | 0.9421 |
| KNN k          | 10    | 0.9421 |
| PCA dims       | 128   | 0.9363 |
| PCA dims       | **256 (default)** | **0.9422** |
| PCA dims       | 512   | 0.9667 |

---

## Inference Latency (CPU — Intel Core i5)

| Component | Params | Memory | Latency |
|-----------|--------|--------|---------|
| CNN-LSTM Encoder | 0.29M | 1.2 MB | 1.91 ms |
| Normalizing Flow | 6.55M | 26.2 MB | 2.50 ms |
| PCA Transform    | —     | 78.2 MB | 2.73 ms |
| KNN Index        | —     | 9.8 MB  | 2.79 ms |
| **Total**        | **7.14M** | **115.4 MB** | **9.93 ms** |

> Suitable for IoT gateway deployment (< 10 ms per packet)

---

## Datasets

| Dataset | Normal | Attack | Type |
|---------|--------|--------|------|
| CICIoT-2023  | BenignTraffic.pcap | DDoS-ICMP Flood | IoT |
| CICIDS2017   | Monday traffic | Multiple attacks | IDS |
| ISCXVPN2016  | NonVPN PCAPs | VPN traffic | Privacy |
| ISCXTor2016  | NonTOR PCAPs | TOR traffic | Anonymity |

---

## Installation

```bash
pip install FrEIA einops scikit-learn scapy
pip install matplotlib seaborn torch
```

---

## Repository Structure

```
Network-Traffic-Analysis/
├── CIC IDS.py              ← CICIDS2017 pipeline
├── CIC IoT                 ← CICIoT-2023 pipeline
├── TOR.py                  ← ISCXTor2016 pipeline
├── VPN.py                  ← ISCXVPN2016 pipeline
├── NLL Explainability.ipynb← Explainability analysis
├── results/
│   ├── cicids_ablation_results.json
│   ├── tor_vpn_ablation.json
│   └── ciciot_ablation.json
├── requirements.txt
└── README.md
```

---

## Citation

```bibtex
@article{dhanya2024cafe,
  title={Semi-Supervised Network Traffic Anomaly Detection
         using Normalizing Flow and KNN Ensemble Scoring},
  author={Dhanya P and Arun Cyril Jose and Ebin Deni Raj},
  journal={***************************},
  year={2024}
}
```

---

> 
