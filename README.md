# protocol-constrained-adversarial-traffic

Protocol-constrained adversarial attacks on image-encoded network traffic classifiers, with defense, explainability, and calibration analysis. Computer Networks research project.

## Research question
Can adversarial perturbations restricted to attacker-controllable protocol fields (padding, timing, packet length) still fool a CNN/ViT classifier on image-encoded traffic, and can an explainability + calibration layer reveal when it is being fooled?

## Repo layout
| Path | Purpose |
|---|---|
| `src/encoding/` | flow -> image encoder and field-to-pixel map |
| `src/data/` | parsing, preprocessing, splits |
| `src/models/` | CNN / ViT classifiers |
| `src/attacks/` | unconstrained and masked FGSM/PGD |
| `src/defenses/` | adversarial training, input purification |
| `src/analysis/` | explainability, calibration |
| `configs/` | seeds, epsilons, split settings |
| `results/` | saved metrics and figures |
| `report/` | Phase 3 document |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data
Raw data is not committed. Download CICIDS2017 (PCAPs + CICFlowMeter CSVs) from the official UNB page and UNSW-NB15 from UNSW, and place them under `data/raw/`.

## Step 1: feasibility check
```bash
python scripts/feasibility_check.py --pcap data/raw/<one-day>.pcap --max-packets 200000
```

## Reproduce our results
_Fill in as experiments are completed: exact commands, in order, with seeds from `configs/default.yaml`._

## Team
- Member 1:
- Member 2:
