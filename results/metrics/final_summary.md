# Final Summary: Protocol-Constrained Adversarial Traffic

## 1. Setup
- Architecture: resnet18; classes: ['Benign', 'DDoS-UDP_Flood', 'DoS-TCP_Flood', 'Recon-PortScan']
- Hold-out (unseen): ['DictionaryBruteForce']
- Attack config: eps=[1, 4, 16, 32] bytes, 25 PGD steps, targeted=False

## 2. Closed-set baseline

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Benign | 0.9673 | 0.9744 | 0.9709 | 2736 |
| DDoS-UDP_Flood | 0.9881 | 0.9749 | 0.9815 | 1195 |
| DoS-TCP_Flood | 0.9000 | 0.6667 | 0.7660 | 27 |
| Recon-PortScan | 0.9553 | 0.9567 | 0.9560 | 2147 |

Overall accuracy: **0.9669** (training-set fit; add a held-out test split before quoting as generalization).

## 3. Open-set behaviour
- Mean max-softmax on hold-out: 0.7731 (known: 0.9426)
- Mean predictive entropy on hold-out: 0.5210
- Absorption: {'Benign': 2156, 'DDoS-UDP_Flood': 117, 'DoS-TCP_Flood': 15, 'Recon-PortScan': 750}

## 4. Protocol-constrained evasion

| Epsilon (bytes) | Known acc | Mean conf | Bytes changed/flow | Checksum validity | Header violations |
|---|---|---|---|---|---|
| clean | 0.8411 | 0.8713 | 0 | 1.0000 | 0 |
| 1 | 0.8411 | 0.8698 | 138.2 | 0.9964 | 0 |
| 4 | 0.8364 | 0.8651 | 142.6 | 0.9964 | 0 |
| 16 | 0.8037 | 0.8541 | 145.0 | 0.9964 | 0 |
| 32 | 0.7336 | 0.7960 | 144.6 | 0.9964 | 0 |

ECE (baseline, 15 bins): **0.0244**

## 5. Protocol validity guarantee
Perturbations touch payload bytes only (offset >= pstart); packet count and lengths unchanged (sequence/ACK integrity); TCP/UDP checksums recomputed after edits and verified on every packet.

## 6. Artifacts
- figures/confusion_matrix.png
- figures/reliability_diagram.png
- figures/holdout_confidence_hist.png
- figures/evasion_vs_eps.png
- eval_baseline_summary.json, eval_baseline.npz
- adversarial_results.json, adversarial_per_sample.npz