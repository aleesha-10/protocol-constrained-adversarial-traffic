"""Aggregate the control experiments over several models (seeds) into one table and figure.

  python scripts/aggregate_controls.py
Reads results/controls/controls_summary_*.csv (one file per --tag). With two or more tags it reports mean and
standard deviation across models; with one tag it just reports the values.
Writes results/controls/controls_table.md (paste into the report) and fig_controls_vs_eps.png.
"""
import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARM_LABEL = {"payload": "payload only (the constrained attack)", "ch0_bytes": "all bytes of channel 0",
             "all_pixels": "all pixels (unconstrained)", "random_payload": "random noise, payload only",
             "random_all": "random noise, all pixels"}
ARM_ORDER = ["payload", "ch0_bytes", "all_pixels", "random_payload", "random_all"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/controls")
    ap.add_argument("--subset", default="has_payload", choices=["has_payload", "all"])
    args = ap.parse_args()
    files = sorted(glob.glob(f"{args.dir}/controls_summary_*.csv"))
    if not files:
        raise SystemExit("no controls_summary_*.csv found; run attack_controls.py first")
    rates = defaultdict(dict)                      # (arm, eps, class) -> {tag: (rate, k, n)}
    for fn in files:
        for r in csv.DictReader(open(fn)):
            if r["kind"] != "flip" or r["subset"] != args.subset or r["rate"] in ("", "nan"):
                continue
            rates[(r["arm"], int(r["eps"]), r["class"])][r["tag"]] = (float(r["rate"]), int(r["flips"]), int(r["n_correct_before"]))
    tags = sorted({t for v in rates.values() for t in v})
    eps_list = sorted({k[1] for k in rates})
    print(f"models (tags): {tags} | subset: {args.subset}")

    def cell(arm, e, c):
        v = rates.get((arm, e, c))
        if not v:
            return "n/a", np.nan, np.nan
        rs = np.array([x[0] for x in v.values()])
        k, n = sum(x[1] for x in v.values()), sum(x[2] for x in v.values())
        txt = f"{100*rs.mean():.1f}%" + (f" ± {100*rs.std(ddof=1):.1f}" if len(rs) > 1 else "") + f" ({k}/{n})"
        return txt, rs.mean(), (rs.std(ddof=1) if len(rs) > 1 else 0.0)

    md = [f"Flip rate by attack arm (flows correct before the attack; {args.subset} flows; models: {', '.join(tags)}). "
          "Cells: mean over models ± standard deviation (flips/flows summed over models).", ""]
    for c in ("DDoS-UDP_Flood", "Recon-PortScan"):
        md += [f"**{c}**", "", "| Attack arm | " + " | ".join(f"ε = {e}" for e in eps_list) + " |", "|---|" + "---|" * len(eps_list)]
        for arm in ARM_ORDER:
            md.append(f"| {ARM_LABEL[arm]} | " + " | ".join(cell(arm, e, c)[0] for e in eps_list) + " |")
        md.append("")
    Path(args.dir, "controls_table.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, c in zip(axes, ("DDoS-UDP_Flood", "Recon-PortScan")):
        for arm in ARM_ORDER:
            m = [cell(arm, e, c)[1] for e in eps_list]; s = [cell(arm, e, c)[2] for e in eps_list]
            ax.errorbar(eps_list, m, yerr=s, marker="o", capsize=3, label=ARM_LABEL[arm], ls="--" if arm.startswith("random") else "-")
        ax.set_xscale("symlog", linthresh=1); ax.set_xticks(eps_list); ax.set_xticklabels([str(e) for e in eps_list])
        ax.set_title(c); ax.set_xlabel("budget epsilon (byte values)"); ax.grid(alpha=0.25)
    axes[0].set_ylabel("flip rate"); axes[1].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Constrained vs unconstrained attack and random-noise controls ({len(tags)} model(s))", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(Path(args.dir) / "fig_controls_vs_eps.png", dpi=200)
    print(f"\nwrote {args.dir}/controls_table.md and fig_controls_vs_eps.png")


if __name__ == "__main__":
    main()
