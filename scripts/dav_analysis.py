"""DAV analysis from the stored per-sample attack results. No GPU and no model needed.

  python scripts/dav_analysis.py
  python scripts/dav_analysis.py --npz results/metrics/adversarial_per_sample_OFFICIAL.npz --out results/dav

Reads the per-sample file written by adversarial_attack.py (keys like '32_adv_pred', '32_adv_conf',
'1_clean_pred', '1_n_changed', ...) and writes figures (PNG), tables (CSV) and dav_summary.json to --out.

What it produces
  fig_metric_trap.png        the script's 'evasion rate' vs the attack-induced flip rate (metric integrity)
  fig_transitions.png        where flows go: clean prediction -> adversarial prediction
  fig_confidence_shift.png   clean vs adversarial confidence per flow, and confidence of flipped flows
  fig_confidently_wrong.png  share of flipped flows still predicted with high confidence
  fig_abstention.png         risk-coverage and catch-vs-false-rejection curves (confidence threshold as a defense)
  fig_capacity.png           flip rate against the number of bytes the attack could change in a flow

Class indices are assumed to be 0 Benign, 1 DDoS-UDP_Flood, 2 DoS-TCP_Flood, 3 Recon-PortScan
(the order in the baseline report). Change NAMES below if your model uses another order.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

NAMES = ["Benign", "DDoS-UDP_Flood", "DoS-TCP_Flood", "Recon-PortScan"]
KNOWN = NAMES[1:]
UNSEEN = "DictionaryBruteForce"
SHORT = {"DDoS-UDP_Flood": "DDoS-UDP", "DoS-TCP_Flood": "DoS-TCP", "Recon-PortScan": "Recon",
         UNSEEN: "BruteForce (unseen)", "Benign": "Benign"}
COLOR = {"DDoS-UDP_Flood": "#1f77b4", "Recon-PortScan": "#d62728", "DoS-TCP_Flood": "#2ca02c", UNSEEN: "#7f7f7f"}
CONF_THR = (0.9, 0.8)


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / den, (c + h) / den


def load(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    eps = sorted({int(k.split("_")[0]) for k in d.files if k.split("_")[0].isdigit()})
    if not eps:
        raise SystemExit("No epsilon-prefixed keys found in " + str(npz_path))
    lab = d[f"{eps[0]}_label"].astype(str)
    D = dict(eps=eps, labels=lab)
    D["lab_idx"] = np.array([NAMES.index(l) if l in KNOWN else -1 for l in lab])
    D["clean_pred"] = d[f"{eps[0]}_clean_pred"].astype(int)
    D["clean_conf"] = d[f"{eps[0]}_clean_conf"].astype(float)
    for e in eps:
        D[(e, "adv_pred")] = d[f"{e}_adv_pred"].astype(int)
        D[(e, "adv_conf")] = d[f"{e}_adv_conf"].astype(float)
        for k in ("n_changed", "valid_rate", "hdr_violations", "realization_diff"):
            if f"{e}_{k}" in d.files:
                D[(e, k)] = d[f"{e}_{k}"].astype(float)
    return D


def outcomes(D, e):
    li, cp, ap = D["lab_idx"], D["clean_pred"], D[(e, "adv_pred")]
    known = li >= 0
    correct0 = known & (cp == li)
    flip = correct0 & (ap != cp)
    to_benign = flip & (ap == 0)
    return known, correct0, flip, to_benign, ap


def footer(fig, D, n_sampled):
    n = len(D["labels"])
    txt = f"{n} of {n_sampled} sampled flows had editable payload and produced a result. " if n_sampled else f"{n} flows with a result. "
    txt += "Flip rate = flows correct before the attack whose prediction changed. One run; intervals are Wilson 95%, flows treated as independent."
    fig.text(0.01, 0.005, txt, fontsize=7, color="#444444", ha="left", va="bottom")


def clean_json(o):
    """Replace NaN/inf (not valid JSON) by None, recursively; convert numpy scalars."""
    if isinstance(o, dict):
        return {k: clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def save_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/metrics/adversarial_per_sample_OFFICIAL.npz")
    ap.add_argument("--json", default="results/metrics/adversarial_results_OFFICIAL.json")
    ap.add_argument("--out", default="results/dav")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    D = load(args.npz)
    eps, lab = D["eps"], D["labels"]
    emax = max(eps)
    n_sampled = None
    if Path(args.json).exists():
        with open(args.json, "r", encoding="utf-8") as f:
            cfg = json.load(f).get("config", {})
        if "max_flows_per_class" in cfg and "attack_classes" in cfg:
            n_sampled = int(cfg["max_flows_per_class"]) * len(cfg["attack_classes"])
    classes = [c for c in (KNOWN + [UNSEEN]) if (lab == c).any()]
    summary = dict(n_rows=int(len(lab)), n_sampled=n_sampled, eps=eps, per_class={}, uncertainty={}, capacity={})

    # ---------------- 1. corrected flip table ----------------
    rows = []
    for c in classes:
        m = lab == c
        for e in eps:
            known, correct0, flip, to_benign, ap_ = outcomes(D, e)
            if c == UNSEEN:
                changed = int((ap_[m] != D["clean_pred"][m]).sum())
                n = int(m.sum())
                lo, hi = wilson(changed, n)
                rows.append([c, e, n, "", changed, changed / n if n else float("nan"), lo, hi, ""])
                continue
            n_ok = int(correct0[m].sum())
            k = int(flip[m].sum())
            lo, hi = wilson(k, n_ok)
            rows.append([c, e, int(m.sum()), n_ok, k, k / n_ok if n_ok else float("nan"), lo, hi, int(to_benign[m].sum())])
            summary["per_class"].setdefault(c, {})[str(e)] = dict(
                n=int(m.sum()), n_correct=n_ok, flips=k, flip_rate=k / n_ok if n_ok else None,
                ci_lo=lo, ci_hi=hi, to_benign=int(to_benign[m].sum()))
    save_csv(out / "flip_table.csv",
             ["class", "eps", "n_with_result", "n_correct_before", "flips_or_changed", "rate", "ci_lo", "ci_hi", "to_benign"], rows)
    summary["per_class"][UNSEEN] = {str(r[1]): dict(n=r[2], prediction_changed=r[4], rate=r[5]) for r in rows if r[0] == UNSEEN}

    # ---------------- 2. metric trap figure ----------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    shown = [c for c in ("DDoS-UDP_Flood", "Recon-PortScan") if (lab == c).sum() >= 30]
    x = [0] + eps
    for c in shown:
        m = lab == c
        li = D["lab_idx"]
        err0 = float((D["clean_pred"][m] != li[m]).mean())
        naive = [err0] + [float((D[(e, "adv_pred")][m] != li[m]).mean()) for e in eps]
        ax[0].plot(x, naive, marker="o", color=COLOR[c], label=f"{SHORT[c]} (n={int(m.sum())})")
        ax[0].axhline(err0, color=COLOR[c], ls=":", lw=1)
        ax[0].text(0, err0 + 0.006, f"error before any attack: {err0:.3f}", color=COLOR[c], fontsize=7, ha="left", va="bottom")
        fr, lo, hi = [], [], []
        for e in eps:
            _, correct0, flip, _, _ = outcomes(D, e)
            k, n_ok = int(flip[m].sum()), int(correct0[m].sum())
            a, b = wilson(k, n_ok)
            fr.append(k / n_ok if n_ok else np.nan); lo.append(a); hi.append(b)
        yerr_lower = np.maximum(0.0, np.array(fr) - np.array(lo))
        yerr_upper = np.array(hi) - np.array(fr)
        ax[1].errorbar(eps, fr, yerr=[yerr_lower, yerr_upper], marker="o", capsize=3,
           color=COLOR[c], label=f"{SHORT[c]} (n correct={int(correct0[m].sum())})")
    ax[0].set_title("What the script calls 'evasion rate'\n(includes errors that existed before the attack)", fontsize=10)
    ax[1].set_title("Attack-induced flip rate\n(only flows classified correctly before the attack)", fontsize=10)
    for a, xt in zip(ax, (x, eps)):
        a.set_xlabel("budget epsilon (byte values)"); a.set_ylim(0, 0.40); a.set_xscale("symlog", linthresh=1)
        a.set_xticks(xt); a.set_xticklabels((["clean"] if xt is x else []) + [str(e) for e in eps]); a.grid(alpha=0.25); a.legend(fontsize=8, loc="upper left")
    ax[1].set_xlim(0.7, max(eps) * 1.15)
    ax[0].set_ylabel("share of flows (same axis in both panels)")
    fig.suptitle("Metric integrity: the script's 'evasion rate' against the attack-induced flip rate", fontsize=11)
    footer(fig, D, n_sampled)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out / "fig_metric_trap.png", dpi=args.dpi); plt.close(fig)

    # ---------------- 3. transitions ----------------
    trows = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for a, c in zip(axes, ["DDoS-UDP_Flood", "Recon-PortScan", UNSEEN]):
        m = lab == c
        M = np.zeros((4, 4), int)
        for cp, ap_ in zip(D["clean_pred"][m], D[(emax, "adv_pred")][m]):
            M[cp, ap_] += 1
        for i in range(4):
            for j in range(4):
                trows.append([c, emax, NAMES[i], NAMES[j], int(M[i, j])])
        im = a.imshow(M, cmap="Blues", vmin=0, vmax=max(M.max(), 1))
        a.set_xticks(range(4)); a.set_yticks(range(4))
        a.set_xticklabels([SHORT[n] for n in NAMES], rotation=30, fontsize=8); a.set_yticklabels([SHORT[n] for n in NAMES], fontsize=8)
        for i in range(4):
            for j in range(4):
                a.text(j, i, M[i, j], ha="center", va="center", fontsize=9, color="white" if M[i, j] > M.max() * 0.55 else "black")
        a.set_xlabel("prediction after the attack"); a.set_ylabel("prediction before the attack")
        a.set_title(f"{SHORT[c]}, eps = {emax} (n = {int(m.sum())})", fontsize=10)
    fig.suptitle("Where flows go under the constrained attack (diagonal = unchanged; rows = original prediction)", fontsize=11)
    footer(fig, D, n_sampled)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(out / "fig_transitions.png", dpi=args.dpi); plt.close(fig)
    save_csv(out / "transitions.csv", ["class", "eps", "clean_pred", "adv_pred", "count"], trows)

    # ---------------- 4. confidence shift ----------------
    pool = np.isin(lab, ["DDoS-UDP_Flood", "Recon-PortScan"])
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
    known, correct0, flip, to_benign, ap_ = outcomes(D, emax)
    cc, ac = D["clean_conf"], D[(emax, "adv_conf")]
    groups = [("correct, unchanged", pool & correct0 & ~flip, "#9aa0a6", "o"),
              ("flipped to Benign", pool & to_benign, "#d62728", "o"),
              ("flipped to another attack class", pool & flip & ~to_benign, "#ff7f0e", "o"),
              ("already wrong before the attack", pool & ~correct0, "#1f77b4", "x")]
    for name, m, col, mk in groups:
        ax[0].scatter(cc[m], ac[m], s=22, c=col, marker=mk, alpha=0.75, label=f"{name} ({int(m.sum())})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set_xlim(0, 1.02); ax[0].set_ylim(0, 1.02)
    ax[0].set_xlabel("confidence before the attack"); ax[0].set_ylabel(f"confidence after the attack (eps = {emax})")
    ax[0].set_title("Per-flow confidence: DDoS-UDP and Recon flows", fontsize=10); ax[0].legend(fontsize=7, loc="lower right"); ax[0].grid(alpha=0.25)
    bins = np.linspace(0, 1, 11)
    cw_rows = []
    for e in eps:
        _, _, fl, _, _ = outcomes(D, e)
        mm = pool & fl
        conf = D[(e, "adv_conf")][mm]
        if len(conf):
            ax[1].hist(conf, bins=bins, histtype="step", lw=1.6, label=f"eps {e} (n flipped = {len(conf)})")
        row = [e, int(mm.sum())]
        for t in CONF_THR:
            row.append(float((conf >= t).mean()) if len(conf) else "")
        row.append(float(np.median(conf)) if len(conf) else "")
        cw_rows.append(row)
        summary["uncertainty"][str(e)] = dict(n_flipped=int(mm.sum()),
                                              share_conf_ge_0_9=(float((conf >= 0.9).mean()) if len(conf) else None),
                                              share_conf_ge_0_8=(float((conf >= 0.8).mean()) if len(conf) else None),
                                              median_conf=(float(np.median(conf)) if len(conf) else None))
    ax[1].axvline(0.9, color="k", ls=":", lw=1); ax[1].set_xlim(0, 1)
    ax[1].set_xlabel("confidence after the attack"); ax[1].set_ylabel("number of flipped flows")
    ax[1].set_title("How confident are the flipped flows?", fontsize=10); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)
    fig.suptitle("Uncertainty under attack", fontsize=11)
    footer(fig, D, n_sampled)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(out / "fig_confidence_shift.png", dpi=args.dpi); plt.close(fig)
    save_csv(out / "confidence_flipped.csv", ["eps", "n_flipped", "share_conf_ge_0.9", "share_conf_ge_0.8", "median_conf"], cw_rows)

    # ---------------- 5. confidently wrong ----------------
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    w = 0.38
    xs = np.arange(len(eps))
    for k, (t, off) in enumerate(zip(CONF_THR, (-w / 2, w / 2))):
        vals = [(r[2 + k] if r[2 + k] != "" else 0) for r in cw_rows]
        b = ax.bar(xs + off, vals, w, label=f"confidence >= {t}")
        for rect, r in zip(b, cw_rows):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.01, f"n={r[1]}", ha="center", fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels([f"eps {e}" for e in eps]); ax.set_ylim(0, 1.1)
    ax.set_ylabel("share of flipped flows"); ax.set_title("Flipped flows that are still predicted with high confidence", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
    footer(fig, D, n_sampled)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "fig_confidently_wrong.png", dpi=args.dpi); plt.close(fig)

    # ---------------- 6. abstention (confidence threshold as a defense) ----------------
    kn = D["lab_idx"] >= 0
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))

    def risk_cov(conf, correct):
        order = np.argsort(-conf)
        corr = correct[order].astype(float)
        k = np.arange(1, len(corr) + 1)
        keep = k >= max(10, int(0.05 * len(corr)))      # the first few flows give an unstable estimate
        return (k / len(corr))[keep], (1 - np.cumsum(corr) / k)[keep]

    cov, risk = risk_cov(D["clean_conf"][kn], (D["clean_pred"] == D["lab_idx"])[kn])
    ax[0].plot(cov, risk, color="#2ca02c", label="no attack")
    sel_rows = []
    for e, col in zip([e for e in eps if e >= 16] or [emax], ["#ff7f0e", "#d62728"]):
        cov, risk = risk_cov(D[(e, "adv_conf")][kn], (D[(e, "adv_pred")] == D["lab_idx"])[kn])
        ax[0].plot(cov, risk, color=col, label=f"attack, eps {e}")
    ax[0].set_xlabel("coverage (share of known-class flows accepted)"); ax[0].set_ylabel("error among accepted flows")
    ax[0].set_title("Risk-coverage: does rejecting low-confidence flows help?", fontsize=10); ax[0].set_ylim(0, 0.5); ax[0].grid(alpha=0.25); ax[0].legend(fontsize=8)
    for e, col in zip([e for e in eps if e >= 16] or [emax], ["#ff7f0e", "#d62728"]):
        _, c0, fl, _, _ = outcomes(D, e)
        pos, neg = D[(e, "adv_conf")][fl], D["clean_conf"][c0]
        if len(pos) >= 3 and len(neg) >= 3:
            y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
            score = 1 - np.r_[pos, neg]
            auc = roc_auc_score(y, score)
            taus = np.unique(np.r_[pos, neg, 0.0, 1.0001])
            tpr = [(pos < t).mean() for t in taus]; fpr = [(neg < t).mean() for t in taus]
            ax[1].plot(fpr, tpr, color=col, label=f"eps {e}: AUROC {auc:.2f} (n flipped = {len(pos)})")
            summary["uncertainty"].setdefault(str(e), {})["auroc_confidence_detector"] = float(auc)
            for t in (0.5, 0.7, 0.9):
                sel_rows.append([e, t, float((pos < t).mean()), float((neg < t).mean())])
        else:
            summary["uncertainty"].setdefault(str(e), {})["auroc_confidence_detector"] = None
    ax[1].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[1].set_xlabel("clean-correct flows wrongly rejected"); ax[1].set_ylabel("flipped flows caught")
    ax[1].set_title("Reject if confidence is below a threshold", fontsize=10); ax[1].grid(alpha=0.25); ax[1].legend(fontsize=8, loc="lower right")
    ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1.02)
    fig.suptitle("Confidence thresholding as a training-free defense (known-class flows)", fontsize=11)
    footer(fig, D, n_sampled)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(out / "fig_abstention.png", dpi=args.dpi); plt.close(fig)
    save_csv(out / "abstention.csv", ["eps", "threshold", "flipped_caught", "clean_correct_rejected"], sel_rows)

    # ---------------- 7. editable capacity vs flip ----------------
    cap_key = (min(eps), "n_changed")
    if cap_key in D:
        cap = D[cap_key]
        _, c0, fl, _, _ = outcomes(D, emax)
        pm = pool & c0
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
        crow = []
        if pm.sum() >= 9:
            q = np.quantile(cap[pm], [1 / 3, 2 / 3])
            edges = [-np.inf, q[0], q[1], np.inf]
            names = ["low", "medium", "high"]
            rates, los, his, ns = [], [], [], []
            for i in range(3):
                mb = pm & (cap > edges[i]) & (cap <= edges[i + 1])
                k, n = int(fl[mb].sum()), int(mb.sum())
                lo, hi = wilson(k, n)
                rates.append(k / n if n else 0); los.append(lo); his.append(hi); ns.append(n)
                crow.append([names[i], float(cap[mb].min()) if n else "", float(cap[mb].max()) if n else "", n, k, rates[-1], lo, hi])
            ax[0].bar(names, rates, yerr=[np.array(rates) - np.array(los), np.array(his) - np.array(rates)], capsize=4, color="#4c72b0")
            for i, n in enumerate(ns):
                ax[0].text(i, his[i] + 0.01, f"n={n}", ha="center", fontsize=8)
            ax[0].set_ylim(0, max(0.5, max(his) + 0.08))
            rho, pval = spearmanr(cap[pm], fl[pm].astype(float))
            summary["capacity"] = dict(spearman_rho=float(rho), p_value=float(pval), n=int(pm.sum()), bins=crow, eps=emax,
                                       capacity_proxy=f"bytes changed per flow at eps={min(eps)}")
            ptxt = "p < 0.001" if pval < 0.001 else f"p = {pval:.3f}"
            ax[0].set_title(f"Flip rate at eps {emax} by editable capacity\nSpearman rho = {rho:.2f} ({ptxt}, n = {int(pm.sum())})", fontsize=10)
        ax[0].set_xlabel("capacity tercile (bytes changed at the smallest budget)"); ax[0].set_ylabel("flip rate"); ax[0].grid(alpha=0.25, axis="y")
        rng = np.random.RandomState(0)
        for c in classes:
            if c == UNSEEN:
                continue                      # a flip is only defined for flows that were correct before the attack
            m = (lab == c) & c0
            if m.sum() == 0:
                continue
            ax[1].scatter(cap[m], rng.uniform(-0.03, 0.03, m.sum()) + (fl[m] * 1.0), s=14, color=COLOR[c], alpha=0.6, label=f"{SHORT[c]} ({int(m.sum())})")
        ax[1].set_yticks([0, 1]); ax[1].set_yticklabels(["not flipped", f"flipped (eps {emax})"])
        ax[1].set_xlabel("bytes changed per flow at the smallest budget")
        ax[1].set_title("Flows correct before the attack: capacity vs outcome", fontsize=10); ax[1].legend(fontsize=7, loc="center right"); ax[1].grid(alpha=0.25)
        fig.suptitle("Is vulnerability explained by how much payload the attacker can edit?", fontsize=11)
        footer(fig, D, n_sampled)
        fig.tight_layout(rect=(0, 0.03, 1, 0.93))
        fig.savefig(out / "fig_capacity.png", dpi=args.dpi); plt.close(fig)
        save_csv(out / "capacity.csv", ["tercile", "min_bytes", "max_bytes", "n", "flips", "flip_rate", "ci_lo", "ci_hi"], crow)

    # ---------------- summary ----------------
    known, correct0, _, _, _ = outcomes(D, emax)
    summary["known_flows"] = int(known.sum())
    summary["correct_before"] = int(correct0.sum())
    with open(out / "dav_summary.json", "w", encoding="utf-8") as f:
        json.dump(clean_json(summary), f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    print(f"Wrote figures, CSVs and dav_summary.json to {out}/")
    print(f"flows with a result: {len(lab)} | known-class: {int(known.sum())} | correct before the attack: {int(correct0.sum())}")
    for c in ("DDoS-UDP_Flood", "Recon-PortScan"):
        r = summary["per_class"].get(c, {}).get(str(emax))
        if r and r["flip_rate"] is not None:
            print(f"  {c}: flip rate at eps {emax} = {r['flips']}/{r['n_correct']} = {r['flip_rate']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}], to Benign {r['to_benign']}")
    u = summary["uncertainty"].get(str(emax), {})
    if u:
        print(f"  flipped flows at eps {emax}: {u.get('n_flipped')} | share with confidence >= 0.9: {u.get('share_conf_ge_0_9')} | AUROC of confidence as detector: {u.get('auroc_confidence_detector')}")
    if summary["capacity"]:
        print(f"  capacity vs flip: Spearman rho = {summary['capacity']['spearman_rho']:.2f} (p = {summary['capacity']['p_value']:.3f}, n = {summary['capacity']['n']})")


if __name__ == "__main__":
    main()
