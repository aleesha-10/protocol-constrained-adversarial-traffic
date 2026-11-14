"""Aggregate baseline + adversarial results into figures and final_summary.md.

Inputs:  results/metrics/eval_baseline_summary.json + .npz   (from evaluate.py)
         results/metrics/adversarial_results.json            (from adversarial_attack.py)
Outputs: results/metrics/figures/*.png and results/metrics/final_summary.md
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

METRICS = Path("results/metrics")
FIGS = METRICS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

base = json.loads((METRICS / "eval_baseline_summary.json").read_text())
adv = json.loads((METRICS / "adversarial_results.json").read_text())
npz = np.load(METRICS / "eval_baseline.npz")
names = [n for n, _ in sorted(base["class_to_idx"].items(), key=lambda kv: kv[1])]

# --- Figure 1: confusion matrix -------------------------------------------
cm = confusion_matrix(npz["y_true"], npz["y_pred"], labels=range(len(names)))
fig, ax = plt.subplots(figsize=(5.4, 4.6), constrained_layout=True)
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right")
ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
for i in range(len(names)):
    for j in range(len(names)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"Closed-set confusion matrix (acc = {base['closed_set_accuracy']:.3f})")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.savefig(FIGS / "confusion_matrix.png", dpi=160); plt.close(fig)

# --- Figure 2: reliability diagram (calibration) ---------------------------
probs, yt = npz["known_probs"], npz["y_true"]
conf, pred = probs.max(1), probs.argmax(1)
correct = (pred == yt).astype(float)
bins = np.linspace(0, 1, 16); idx = np.digitize(conf, bins) - 1
ece, xs, ys = 0.0, [], []
for b in range(15):
    s = idx == b
    if s.sum():
        xs.append(conf[s].mean()); ys.append(correct[s].mean())
        ece += s.mean() * abs(correct[s].mean() - conf[s].mean())
fig, ax = plt.subplots(figsize=(4.6, 4.6), constrained_layout=True)
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
ax.bar(xs, ys, width=0.062, edgecolor="k", alpha=0.8, label="model")
ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
ax.set_title(f"Reliability diagram (ECE = {ece:.4f})"); ax.legend()
fig.savefig(FIGS / "reliability_diagram.png", dpi=160); plt.close(fig)

# --- Figure 3: confidence on known vs hold-out -----------------------------
hold = npz["holdout_conf"]
fig, ax = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
ax.hist(conf, bins=40, alpha=0.6, density=True, label="known classes")
ax.hist(hold, bins=40, alpha=0.6, density=True,
        label="hold-out: " + ", ".join(base["holdout_classes"]))
ax.set_xlabel("Max-softmax confidence"); ax.set_ylabel("Density"); ax.legend()
ax.set_title("Confidence: known vs unseen traffic")
fig.savefig(FIGS / "holdout_confidence_hist.png", dpi=160); plt.close(fig)

# --- Figure 4: evasion vs epsilon -------------------------------------------
eps_list = sorted(int(e) for e in adv["eps"])
accs = [adv["eps"][str(e)]["known_acc"] for e in eps_list]
confs = [adv["eps"][str(e)]["mean_conf"] for e in eps_list]
fig, ax = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
ax.plot(eps_list, accs, "o-", label="adversarial accuracy (known classes)")
ax.axhline(adv["clean"]["known_acc"], ls="--", c="gray", label="clean accuracy")
ax2 = ax.twinx()
ax2.plot(eps_list, confs, "s--", c="tab:red", label="mean adv confidence")
ax2.set_ylabel("Mean confidence", color="tab:red")
ax.set_xlabel("Epsilon (payload byte-value budget)"); ax.set_ylabel("Accuracy")
ax.set_title("Evasion strength vs perturbation budget")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8)
fig.savefig(FIGS / "evasion_vs_eps.png", dpi=160); plt.close(fig)

# --- final_summary.md --------------------------------------------------------
rep = base["closed_set_report"]
lines = ["# Final Summary: Protocol-Constrained Adversarial Traffic", "",
         "## 1. Setup",
         f"- Architecture: {base['arch']}; classes: {names}",
         f"- Hold-out (unseen): {base['holdout_classes']}",
         f"- Attack config: eps={adv['config']['eps_bytes']} bytes, "
         f"{adv['config']['steps']} PGD steps, targeted={adv['config']['targeted']}", "",
         "## 2. Closed-set baseline", "",
         "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
for c in names:
    r = rep[c]
    lines.append(f"| {c} | {r['precision']:.4f} | {r['recall']:.4f} | "
                 f"{r['f1-score']:.4f} | {int(r['support'])} |")
lines += ["", f"Overall accuracy: **{base['closed_set_accuracy']:.4f}** (training-set fit; "
          "add a held-out test split before quoting as generalization).", "",
          "## 3. Open-set behaviour",
          f"- Mean max-softmax on hold-out: {base['holdout_mean_conf']:.4f} "
          f"(known: {float(conf.mean()):.4f})",
          f"- Mean predictive entropy on hold-out: {base['holdout_mean_entropy']:.4f}",
          f"- Absorption: {base['holdout_pred_distribution']}", "",
          "## 4. Protocol-constrained evasion", "",
          "| Epsilon (bytes) | Known acc | Mean conf | Bytes changed/flow | "
          "Checksum validity | Header violations |", "|---|---|---|---|---|---|",
          f"| clean | {adv['clean']['known_acc']:.4f} | {adv['clean']['mean_conf']:.4f} "
          f"| 0 | 1.0000 | 0 |"]
for e in eps_list:
    s = adv["eps"][str(e)]
    lines.append(f"| {e} | {s['known_acc']:.4f} | {s['mean_conf']:.4f} | "
                 f"{s['mean_bytes_changed']:.1f} | {s['mean_valid_rate']:.4f} | "
                 f"{s['header_violations']} |")
lines += ["", f"ECE (baseline, 15 bins): **{ece:.4f}**", "",
          "## 5. Protocol validity guarantee",
          "Perturbations touch payload bytes only (offset >= pstart); packet count "
          "and lengths unchanged (sequence/ACK integrity); TCP/UDP checksums "
          "recomputed after edits and verified on every packet.", "",
          "## 6. Artifacts", "- figures/confusion_matrix.png",
          "- figures/reliability_diagram.png",
          "- figures/holdout_confidence_hist.png",
          "- figures/evasion_vs_eps.png",
          "- eval_baseline_summary.json, eval_baseline.npz",
          "- adversarial_results.json, adversarial_per_sample.npz"]
(METRICS / "final_summary.md").write_text("\n".join(lines))
print(f"ECE: {ece:.4f}")
print(f"Saved 4 figures to {FIGS} and {METRICS/'final_summary.md'}")