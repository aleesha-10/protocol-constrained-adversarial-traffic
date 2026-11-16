# Add-ons: DAV analysis, evidence dashboard, claim check

All three scripts read files you already have. No GPU, no model, no PCAPs, no flow cache.
Run them from the project root, on any laptop.

## 1. Install (only if missing)
    pip install numpy scipy scikit-learn matplotlib

## 2. Copy into the repo
Copy the three files in `scripts/` into your repo's `scripts/` folder.
They expect these files (they are already in your repo):
    results/metrics/adversarial_per_sample_OFFICIAL.npz
    results/metrics/adversarial_results_OFFICIAL.json
    results/metrics/eval_baseline_summary.json        (claim_check only, optional)

## 3. Run, in this order
    python scripts/claim_check.py --baseline-json results/metrics/eval_baseline_summary.json
    python scripts/dav_analysis.py
    python scripts/build_dashboard.py

1. claim_check.py prints PASS / FAIL / SKIPPED for every number quoted in the Phase 3 report.
   Fix every FAIL either in the report or, if the report is right and the definition differs, in the CLAIMS list at
   the top of the script. A FAIL is a finding, not a crash: it means the report and the data disagree.
2. dav_analysis.py writes to results/dav/: six figures (fig_*.png), CSV tables and dav_summary.json.
3. build_dashboard.py writes results/dav/dashboard.html. Double-click it. It works offline.

## 4. What each figure is for (and where it goes in the report)
| File | Question | Report section |
|---|---|---|
| fig_metric_trap.png | Does the script's 'evasion rate' overstate the attack? | 8.5 and 9 (metric integrity) |
| fig_transitions.png | Where do flows go after the attack? | 9.3 |
| fig_confidence_shift.png | How confident are flipped flows? | 10 |
| fig_confidently_wrong.png | Share of flipped flows still predicted with confidence >= 0.9 | 10 (closes the PENDING item) |
| fig_abstention.png | Can a confidence threshold catch the attack without retraining? | 10 / new subsection |
| fig_capacity.png | Does editable payload size explain who flips (and DoS-TCP's immunity)? | 9.5 |
| dashboard.html | Interactive version of all of the above, with the denominators always visible | DAV deliverable (screenshots) |

## 5. Assumptions to check once
- Class indices are 0 Benign, 1 DDoS-UDP_Flood, 2 DoS-TCP_Flood, 3 Recon-PortScan (order of the baseline report).
  If your model uses another order, change NAMES at the top of dav_analysis.py and claim_check.py, and
  CLASSES/NAMES in build_dashboard.py. A quick check: the console summary of dav_analysis.py should print
  DDoS-UDP 10/94 and Recon 13/86 at eps 32. If it does not, the order is wrong.
- Editable capacity is measured as 'bytes changed per flow at the smallest budget' (key 1_n_changed).
  It is a proxy, not a direct count of payload bytes. Say so in the report.
- Results of the confidence analyses are unknown until you run them. Report whatever they show, including
  'a confidence threshold does not help'.

## 6. Commit
    git add scripts/claim_check.py scripts/dav_analysis.py scripts/build_dashboard.py results/dav
    git status
    git commit -m "Add DAV analysis, evidence dashboard and claim check"
    git push
