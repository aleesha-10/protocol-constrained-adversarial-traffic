# protocol-constrained-adversarial-traffic

Protocol-constrained adversarial attacks on image-encoded network traffic classifiers, with packet-level realization, side-channel evasion (UDP padding & inter-packet timing), protocol-aware adversarial training, input attribution explainability, and calibration analysis.

## Research Question

Can adversarial perturbations restricted to attacker-controllable protocol regions—valid L4 payload bytes (Channel 0), zero-padded packet lengths (Channel 1), and non-negative inter-arrival delays (Channel 2)—evade a ResNet-18 classifier on image-encoded network flows while maintaining zero header invariant violations (`hdr_violations = 0`) on the wire?

## Key Findings Summary

* **Leak-Free Flow Splitting:** Enforcing a unified $\ge 4$-packet flow definition and a chronological 70/15/15 flow-level split on **CIC IoT 2023** yields **86.91% test accuracy** (**0.7222 Macro-F1**, **0.0497 ECE**), correcting the +9.78 percentage point inflation caused by window-level splitting.
* **Payload-Only PGD & Cross-Seed Transfer:** Clamped-input $\varepsilon=32$ PGD on Channel 0 payload pixels produces 10.6%–15.1% white-box flip rates with low cross-seed transferability, while maintaining 100% packet-space realization (`realization_max = 0.0`, `hdr_violations = 0`).
* **Side-Channel Evasion (Padding & Timing):** Exploiting Channels 1 and 2 via protocol-valid UDP zero-padding (`pad_only_1024B`) flips **92.3%** of `DDoS-UDP_Flood` flows, and bounded inter-packet delays (`timing_only_1000ms`) flip **100%** of `DDoS-UDP_Flood` (84.6% directly to `Benign`), **100%** of `DoS-TCP_Flood`, and **57.3%** of `Recon-PortScan`—all with `valid_rate = 1.000` and `hdr_violations = 0`.
* **Protocol-Aware Adversarial Training & Explainability:** Training on masked payload PGD plus side-channel length/timing jitter raises clean held-out test accuracy to **90.18%** (**0.7940 Macro-F1**), while Gradient $\times$ Input attribution shows why side-channel vulnerability persists: the classifier still assigns 30%–40% attribution mass to inter-arrival timing (`Ch2_Time`) and IP header length fields.

## Repository Layout

| Path | Purpose |
|---|---|
| `src/encoding/` | 3-channel flow-to-image encoder (`16 × 64`: raw bytes, packet length, log-IAT) & `FIELD_MAP` |
| `src/data/` | PCAP flow parser ($\ge 4$ packets/flow) and chronological 70/15/15 flow splitter |
| `scripts/` | Training, evaluation, clamped-input PGD controls, cross-seed transfer, length/timing attacks, adversarial training, attribution, DAV plots, and `claim_check.py` |
| `configs/` | Seeds, epsilon budgets, and split parameters |
| `results/metrics/` | Held-out evaluation JSON/NPZ summaries for standard and adversarially trained models |
| `results/controls/` | Clamped-input PGD control metrics and $4 \times 4$ cross-seed transfer matrix |
| `results/length_timing/` | Packet-realizable UDP padding & inter-packet timing attack summaries |
| `results/explainability/` | Gradient $\times$ Input regional attribution shares (`standard` vs. `adv_trained`) |
| `results/dav_new/` | Publication figures for transferability, side-channel evasion, attribution, and pipeline progression |
| `notes/` | Dependency lists (`requirements.txt`, `requirements-analysis.txt`) and project notes |

## Setup

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

pip install -r notes/requirements.txt
pip install -r notes/requirements-analysis.txt
```

## Dataset

Raw PCAPs and trained `.pth` checkpoints are excluded via `.gitignore`. Place the CIC IoT 2023 PCAP captures under `data/raw/<class>/` using the following class folders:

**Known training/validation/test classes:**

- `Benign`
- `DDoS-UDP_Flood`
- `DoS-TCP_Flood`
- `Recon-PortScan`

**Held-out out-of-distribution class:**

- `DictionaryBruteForce`

## Reproducing the Pipeline

Run from the repository root (seed = `42`):

```bash
# 1. Train baseline ResNet-18 on leak-free 70/15/15 chronological flow split
python scripts/train.py

# 2. Evaluate baseline held-out test & OOD metrics
python scripts/evaluate.py

# 3. Run protocol-constrained payload PGD attacks & per-class analysis
python scripts/adversarial_attack.py
python scripts/analyze_per_class.py
python scripts/flip_table.py

# 4. Run clamped-input PGD controls & 4x4 cross-seed transfer matrix
python scripts/attack_controls.py
python scripts/transfer.py

# 5. Run packet-realizable UDP padding & inter-packet timing side-channel attacks
python scripts/length_timing_attack.py

# 6. Compute Gradient x Input channel/region attribution
python scripts/explainability.py

# 7. Train protocol-aware adversarially defended model & re-evaluate robustness/attribution
python scripts/train_adv.py

# 8. Generate DAV figures and audit all numerical claims
python scripts/plot_new_results.py
python scripts/claim_check.py
```

## Team

- **Aleesha**
- **Tayyaba**
