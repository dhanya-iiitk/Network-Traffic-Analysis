# Network Traffic Anomaly Detection
## CAFE-FKESE Framework

A semi-supervised network anomaly detection 
framework operating directly on raw packet 
bytes without feature engineering, trained 
exclusively on normal traffic.

## Framework Architecture

 CNN-LSTM Autoencoding Feature Extractor  
  
  
Flow-KNN Ensemble Scoring Engine  
  Combines Normalizing Flow density estimation 
  (w=0.9) with KNN distance scoring (w=0.1)

Youden-Optimal Binary Decision Module  
  Data-driven threshold using Youden's J statistic

## Results

| Script | Dataset | AUROC |
|--------|---------|-------|
| ciciot2023.py | CICIoT-2023 | 0.9828 |
| cicids2017.py | CICIDS2017  | 0.9655 |
| vpn.py        | ISCXVPN2016 | 0.9376 |
| tor.py        | ISCXTor2016 | 0.8899 |

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

```bash
python cicids2017.py
python ciciot2023.py
python vpn.py
python tor.py
```

## Datasets

Download from University of New Brunswick (UNB):
- [CICIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
- [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)
- [ISCXVPN2016](https://www.unb.ca/cic/datasets/vpn.html)
- [ISCXTor2016](https://www.unb.ca/cic/datasets/tor.html)

## Citation


## Author

 IIITK
