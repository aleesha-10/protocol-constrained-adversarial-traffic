# Final Summary: Protocol-Constrained Adversarial Traffic

## 1. Setup
- Architecture: resnet18; classes: Benign, DDoS-UDP_Flood, DoS-TCP_Flood, Recon-PortScan
- Hold-out (unseen) class: DictionaryBruteForce
- Attack: protocol-constrained PGD, eps = 1, 4, 16, 32 bytes, 25 steps, untargeted, seed 42,
  150 sampled flows per class

## 2. Closed-set baseline

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Benign | 0.9673 | 0.9744 | 0.9709 | 2736 |
| DDoS-UDP_Flood | 0.9881 | 0.9749 | 0.9815 | 1195 |
| DoS-TCP_Flood | 0.9000 | 0.6667 | 0.7660 | 27 |
| Recon-PortScan | 0.9553 | 0.9567 | 0.9560 | 2147 |

Overall accuracy 0.9669. These figures are a training-set fit and must not be read as
generalization. They are also not reconciled with the attack data: on the flows that were
attacked, clean accuracy was 0.790 (DDoS-UDP) and 0.915 (Recon-PortScan), against recalls of
0.975 and 0.957 here. DoS-TCP rests on a support of 27. ECE = 0.0244 (15 bins); the data it was
computed on should be stated before it is quoted.

## 3. Open-set behaviour
- Mean max-softmax on the hold-out class: 0.7731 (known classes: 0.9426).
- Mean predictive entropy on the hold-out class: 0.5210 (units as defined in the evaluation script).
- Where the 3,038 held-out samples were assigned: Benign 2,156 (71%), Recon-PortScan 750 (25%),
  DDoS-UDP_Flood 117 (4%), DoS-TCP_Flood 15 (0.5%). Most of the unseen attack is called Benign.

## 4. Protocol-constrained evasion
Single run (seed 42, 150 sampled flows per class, 25 PGD steps, untargeted). Only 318 of 600
sampled flows produced results; 282 were skipped, including 149 of 150 DoS-TCP flows (cause not
confirmed). Flip rate = share of flows classified correctly before the attack whose prediction
changed afterwards. Intervals are Wilson 95% and treat flows as independent, so they understate
the true uncertainty.

| Class | Clean-correct / n | eps=1 | eps=4 | eps=16 | eps=32 (95% CI) | To Benign at eps=32 |
|---|---|---|---|---|---|---|
| DDoS-UDP_Flood | 94 / 119 | 0 | 0 | 2 (2.1%) | 10 (10.6%, 5.9-18.5) | 3 |
| Recon-PortScan | 86 / 94 | 0 | 1 (1.2%) | 6 (7.0%) | 13 (15.1%, 9.1-24.2) | 12 |
| DoS-TCP_Flood | 0 / 1 | not assessable (n = 1) | | | | |

Known-class accuracy over the 214 known-class flows that produced results: 0.8411 clean,
0.8411 / 0.8364 / 0.8037 / 0.7336 at eps = 1 / 4 / 16 / 32. DictionaryBruteForce (unseen,
n = 104) has no correct clean label, so its prediction changes are not an evasion rate and are
not compared with the known classes. The two classes' intervals overlap at eps = 32, so the
data do not show that one class is more vulnerable than the other. Budgets above 32 bytes were
not evaluated.

## 5. Protocol validity
Edits are restricted to payload bytes (offset >= pstart), packet count and lengths are
unchanged, and checksums are recomputed and verified. 99.64% of packets pass checksum
verification, header violations are 0, and re-parsing reproduces the edited bytes exactly
(max difference 0.0000). The 0.36% failures were not diagnosed.

## 6. Limitations
- Single run (seed 42) on 150 sampled flows per class; 282 of 600 sampled flows produced no
  result, including 149 of 150 DoS-TCP flows, and the cause is unconfirmed.
- Baseline metrics in Section 2 reflect training-set fit, not generalization.
- Confidence intervals treat flows as independent and understate the uncertainty.
- No unconstrained-attack baseline, defense, ablation or cross-dataset test was run, so we make
  no claim that the protocol constraint reduces attack success compared with an unconstrained
  attacker.

## 7. Artifacts
- results/metrics/figures/: confusion_matrix.png, reliability_diagram.png,
  holdout_confidence_hist.png, evasion_vs_eps.png
- results/metrics/: eval_baseline_summary.json, eval_baseline.npz,
  adversarial_results_OFFICIAL.json, adversarial_per_sample_OFFICIAL.npz
- scripts/flip_table.py (per-class flip table with intervals)