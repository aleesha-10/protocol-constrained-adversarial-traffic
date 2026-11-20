# Save as scripts/plot_new_results.py
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Rich Dark Emerald / Maroon / Gold / Plum DAV Palette
BG_DARK = "#0E1111"
PANEL_DARK = "#151A1A"
GRID_COL = "#263030"
TEXT_LIGHT = "#F4F1EA"
TEXT_MUTED = "#A8A39D"

EMERALD = "#1B5E4B"
EMERALD_BRIGHT = "#2E9E7B"
MAROON = "#7A1C2E"
MAROON_BRIGHT = "#C23B56"
PLUM = "#582C5F"
GOLD = "#D4A359"
SLATE = "#4A607A"

plt.rcParams.update({
    "figure.facecolor": BG_DARK,
    "axes.facecolor": PANEL_DARK,
    "axes.edgecolor": GRID_COL,
    "axes.labelcolor": TEXT_LIGHT,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "text.color": TEXT_LIGHT,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "savefig.facecolor": BG_DARK,
    "savefig.dpi": 300,
})


def plot_transfer_matrix(out_dir: Path):
    csv_path = Path("results/controls/transfer_matrix_eps32.csv")
    if not csv_path.exists():
        return
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    seeds = list(dict.fromkeys(r["source"] for r in rows))
    n = len(seeds)
    seed_idx = {seed: i for i, seed in enumerate(seeds)}
    mat = np.full((n, n), np.nan)
    labels = [[""] * n for _ in range(n)]
    for r in rows:
        i, j = seed_idx[r["source"]], seed_idx[r["target"]]
        mat[i, j] = float(r["flip_rate"])
        labels[i][j] = f"{mat[i, j]*100:.1f}%"

    cmap = LinearSegmentedColormap.from_list(
        "emerald_maroon", ["#132421", EMERALD, GOLD, MAROON, MAROON_BRIGHT]
    )

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=0.30)

    for i in range(n):
        for j in range(n):
            is_diag = (i == j)
            ax.text(
                j, i,
                f"{labels[i][j]}\n({'White-box' if is_diag else 'Transfer'})",
                ha="center", va="center",
                color=TEXT_LIGHT if mat[i, j] > 0.08 or is_diag else TEXT_MUTED,
                fontweight="bold" if is_diag else "normal",
                fontsize=9.5,
            )
            if is_diag:
                rect = plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, edgecolor=GOLD, lw=2.0)
                ax.add_patch(rect)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"Target:\n{s}" for s in seeds], fontsize=9.5)
    ax.set_yticklabels([f"Source: {s}" for s in seeds], fontsize=9.5)
    ax.set_title("Cross-Seed Adversarial Transfer Matrix (Payload PGD, ε = 32)", fontsize=12, fontweight="bold", pad=14)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Flow Flip Rate", color=TEXT_LIGHT)
    cbar.ax.yaxis.set_tick_params(color=TEXT_MUTED)

    plt.tight_layout()
    fig.savefig(out_dir / "fig_dav_transfer_matrix.png")
    plt.close(fig)


def plot_sidechannel_dumbbell(out_dir: Path):
    std_path = Path("results/length_timing/length_timing_summary.csv")
    adv_path = Path("results/length_timing/length_timing_summary_adv_trained.csv")
    if not std_path.exists():
        return

    with open(std_path) as f:
        std_rows = list(csv.DictReader(f))
    adv_map = {}
    if adv_path.exists():
        with open(adv_path) as f:
            for r in csv.DictReader(f):
                adv_map[(r["arm"], r["class"])] = float(r["flip_rate"])

    arms = [
        "pad_only_256B",
        "pad_only_1024B",
        "timing_only_100ms",
        "timing_only_1000ms",
        "joint_pad1024B_delay1000ms",
    ]
    arm_labels = [
        "UDP Pad (256B)",
        "UDP Pad (1024B)",
        "Timing Delay (100ms)",
        "Timing Delay (1000ms)",
        "Joint Pad + Delay",
    ]
    classes = ["DDoS-UDP_Flood", "Recon-PortScan", "DoS-TCP_Flood"]
    cls_colors = {
        "DDoS-UDP_Flood": MAROON_BRIGHT,
        "Recon-PortScan": EMERALD_BRIGHT,
        "DoS-TCP_Flood": GOLD,
    }

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2), sharey=True)
    y_pos = np.arange(len(arms))

    for ax, cls in zip(axes, classes):
        ax.grid(axis="x", color=GRID_COL, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        for idx, arm in enumerate(arms):
            row = next((r for r in std_rows if r["arm"] == arm and r["class"] == cls), None)
            if not row:
                continue
            std_rate = float(row["flip_rate"]) * 100.0
            to_ben = int(row["to_benign"])
            adv_rate = adv_map.get((arm, cls), None)

            if adv_rate is not None:
                adv_pct = adv_rate * 100.0
                ax.plot([std_rate, adv_pct], [idx, idx], color=TEXT_MUTED, lw=2.0, alpha=0.75, zorder=2)
                ax.scatter([adv_pct], [idx], color=PLUM, edgecolor=TEXT_LIGHT, s=95, marker="D", zorder=4, label="Adv-Trained" if (idx == 1 and cls == classes[0]) else "")

            ax.scatter([std_rate], [idx], color=cls_colors[cls], edgecolor=TEXT_LIGHT, s=110, zorder=5, label="Standard Model" if (idx == 0 and cls == classes[0]) else "")
            tag = f"{std_rate:.1f}%" + (f" ({to_ben}→Ben)" if to_ben > 0 else "")
            ax.text(min(std_rate + 2.5, 96), idx + 0.16, tag, fontsize=8.2, color=TEXT_LIGHT)

        ax.set_title(cls, fontsize=11, fontweight="bold", color=cls_colors[cls], pad=10)
        ax.set_xlim(-4, 114)
        ax.set_xlabel("Attack Flip Rate (%) [0 Header Violations]")

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(arm_labels, fontsize=9.5, fontweight="bold")
    axes[0].invert_yaxis()

    fig.suptitle(
        "Protocol-Valid Side-Channel Evasion: Standard vs. Adversarially Trained Classifier",
        fontsize=13, fontweight="bold", y=0.98
    )
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=MAROON_BRIGHT, markersize=9, label="Standard Model (Circle)"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=PLUM, markersize=8, label="Adv-Trained Model (Diamond)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=True, facecolor=PANEL_DARK, edgecolor=GRID_COL)
    plt.tight_layout(rect=[0, 0.07, 1, 0.93])
    fig.savefig(out_dir / "fig_dav_sidechannel_dumbbell.png")
    plt.close(fig)


def plot_attribution_composition_and_radar(out_dir: Path):
    std_p = Path("results/explainability/channel_attribution_standard.csv")
    adv_p = Path("results/explainability/channel_attribution_adv_trained.csv")
    if not std_p.exists() or not adv_p.exists():
        return

    with open(std_p) as f:
        std_rows = list(csv.DictReader(f))
    with open(adv_p) as f:
        adv_rows = list(csv.DictReader(f))

    regions = ["ch0_ip_hdr", "ch0_l4_hdr", "ch0_payload", "ch1_length", "ch2_timing"]
    region_labels = ["Ch0: IP Header", "Ch0: L4 Header", "Ch0: Payload", "Ch1: Packet Len", "Ch2: IAT Timing"]
    palette = [SLATE, EMERALD, GOLD, MAROON, PLUM]

    fig = plt.figure(figsize=(14, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0])

    # Left: Paired 100% Horizontal Stacked Composition Bars
    ax1 = fig.add_subplot(gs[0])
    y_labels = []
    data_matrix = []
    for sr, ar in zip(std_rows, adv_rows):
        cls = sr["class"]
        y_labels.append(f"{cls}\n(Standard)")
        data_matrix.append([float(sr[k]) * 100 for k in regions])
        y_labels.append(f"{cls}\n(Adv-Trained)")
        data_matrix.append([float(ar[k]) * 100 for k in regions])

    data_matrix = np.array(data_matrix)
    y_idx = np.arange(len(y_labels))
    left = np.zeros(len(y_labels))

    for col_i, (r_lbl, col_hex) in enumerate(zip(region_labels, palette)):
        vals = data_matrix[:, col_i]
        ax1.barh(y_idx, vals, left=left, height=0.68, color=col_hex, edgecolor=BG_DARK, linewidth=1.2, label=r_lbl)
        for row_i, v in enumerate(vals):
            if v >= 11.0:
                ax1.text(left[row_i] + v / 2.0, row_i, f"{v:.0f}%", ha="center", va="center", fontsize=8, fontweight="bold", color=TEXT_LIGHT)
        left += vals

    ax1.set_yticks(y_idx)
    ax1.set_yticklabels(y_labels, fontsize=8.5)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Share of Gradient × Input Attribution (%)")
    ax1.set_title("Feature Attribution Shift: Standard vs. Adv-Trained", fontsize=11.5, fontweight="bold", pad=10)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=True, facecolor=PANEL_DARK, edgecolor=GRID_COL, fontsize=8.5)

    # Right: Polar Radar of Mean Attribution Across All Classes
    ax2 = fig.add_subplot(gs[1], polar=True)
    ax2.set_facecolor(PANEL_DARK)
    angles = np.linspace(0, 2 * np.pi, len(regions), endpoint=False).tolist()
    angles += angles[:1]

    std_mean = [np.mean([float(r[k]) for r in std_rows]) * 100 for k in regions] + [np.mean([float(r[regions[0]]) for r in std_rows]) * 100]
    adv_mean = [np.mean([float(r[k]) for r in adv_rows]) * 100 for k in regions] + [np.mean([float(r[regions[0]]) for r in adv_rows]) * 100]

    ax2.plot(angles, std_mean, color=MAROON_BRIGHT, lw=2.3, label="Standard (Mean)")
    ax2.fill(angles, std_mean, color=MAROON, alpha=0.28)
    ax2.plot(angles, adv_mean, color=EMERALD_BRIGHT, lw=2.3, label="Adv-Trained (Mean)")
    ax2.fill(angles, adv_mean, color=EMERALD, alpha=0.28)

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(["IP Hdr", "L4 Hdr", "Payload", "Ch1 Len", "Ch2 Time"], fontsize=9.5, fontweight="bold", color=TEXT_LIGHT)
    ax2.yaxis.grid(True, color=GRID_COL, linestyle="--")
    ax2.xaxis.grid(True, color=GRID_COL)
    ax2.set_title("Mean Attribution Profile Across Protocol Channels", fontsize=11.5, fontweight="bold", pad=18)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=True, facecolor=PANEL_DARK, edgecolor=GRID_COL, fontsize=8.5)

    plt.tight_layout()
    fig.savefig(out_dir / "fig_dav_attribution_profile.png", bbox_inches="tight")
    plt.close(fig)


def plot_pipeline_progression(out_dir: Path):
    stages = [
        "1. Leaked Window Split\n(Phase 3 Baseline)",
        "2. Leak-Free Flow Split\n(70/15/15 Chronological)",
        "3. Protocol-Aware\nAdv-Trained Defense",
    ]
    overall_acc = [96.69, 86.91, 90.18]
    macro_f1 = [91.86, 72.22, 79.40]
    dos_tcp_f1 = [76.60, 26.67, 47.06]
    recon_f1 = [95.60, 78.22, 83.27]

    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.grid(True, color=GRID_COL, linestyle="--", alpha=0.6)

    series = [
        ("Overall Accuracy (%)", overall_acc, GOLD, "o", "-"),
        ("Macro-F1 (%)", macro_f1, EMERALD_BRIGHT, "s", "-"),
        ("Recon-PortScan F1 (%)", recon_f1, SLATE, "^", "--"),
        ("DoS-TCP_Flood F1 (%)", dos_tcp_f1, MAROON_BRIGHT, "D", "-"),
    ]
    for label, vals, col, mk, ls in series:
        ax.plot(x, vals, color=col, marker=mk, linestyle=ls, lw=2.5, markersize=8.5, label=label)
        for xi, yi in zip(x, vals):
            ax.text(xi, yi + 2.2, f"{yi:.1f}%", ha="center", fontsize=8.5, fontweight="bold", color=col)

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=9.5, fontweight="bold")
    ax.set_ylim(15, 106)
    ax.set_ylabel("Held-Out Score (%)")
    ax.set_title("Methodological Trajectory: Leakage Removal & Protocol-Aware Defense", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower left", frameon=True, facecolor=PANEL_DARK, edgecolor=GRID_COL, fontsize=9)

    plt.tight_layout()
    fig.savefig(out_dir / "fig_dav_pipeline_trajectory.png")
    plt.close(fig)


def main():
    out_dir = Path("results/dav_new")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_transfer_matrix(out_dir)
    plot_sidechannel_dumbbell(out_dir)
    plot_attribution_composition_and_radar(out_dir)
    plot_pipeline_progression(out_dir)
    print(f"Saved all 4 dark-emerald/maroon DAV figures to {out_dir}")


if __name__ == "__main__":
    main()