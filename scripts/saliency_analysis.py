"""Where does the classifier look? Field-aligned saliency, and whether it predicts which flows flip.

  python scripts/saliency_analysis.py
  python scripts/saliency_analysis.py --ckpt results/metrics/resnet18_seed0.pth --tag seed0

For every sampled flow it takes the input-gradient saliency of the predicted class and splits the saliency mass
into the regions of the traffic image:
  ch0_header   channel 0, bytes of real packets before the payload (IP and TCP/UDP header)
  ch0_payload  channel 0, payload bytes  <- the only region the constrained attack may edit
  ch0_unused   channel 0, positions beyond the packet length or in unused rows
  ch1_length   channel 1 (packet length)      ch2_timing  channel 2 (inter-arrival time)
'enrichment' = share of saliency mass / share of pixels: above 1 means the model leans on that region.
Honest note: input-gradient saliency is post-hoc and its faithfulness is not guaranteed.
Writes saliency_region_mass_<tag>.csv, fig_saliency_regions_<tag>.png, fig_saliency_by_byte_<tag>.png
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from attack_controls import load_model, predict
from audit_utils import ATTACK_CLASSES, MODEL_PATH, P, B, flow_to_image, load_cache, official_flow_list, prepare, select_flows

REGIONS = ["ch0_header", "ch0_payload", "ch0_unused", "ch1_length", "ch2_timing"]
SHORT = {"DDoS-UDP_Flood": "DDoS-UDP", "DoS-TCP_Flood": "DoS-TCP", "Recon-PortScan": "Recon", "DictionaryBruteForce": "BruteForce (unseen)"}


def saliency(model, x, cls):
    x = x.clone().requires_grad_(True)
    logits = model(x)
    g, = torch.autograd.grad(logits.gather(1, cls[:, None]).sum(), x)
    return g.abs()


def region_masks(pr):
    ch0 = np.zeros((3, P, B), bool); ch0[0] = True
    m = {"ch0_header": ch0 & np.stack([pr["any_byte"] & ~pr["payload"]] * 3),
         "ch0_payload": ch0 & np.stack([pr["payload"]] * 3),
         "ch0_unused": ch0 & np.stack([~pr["any_byte"]] * 3)}
    c1 = np.zeros((3, P, B), bool); c1[1] = True
    c2 = np.zeros((3, P, B), bool); c2[2] = True
    m["ch1_length"], m["ch2_timing"] = c1, c2
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=MODEL_PATH)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--official", default="results/metrics/adversarial_per_sample_OFFICIAL.npz")
    ap.add_argument("--out", default="results/saliency")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, c2i = load_model(args.ckpt, device)
    _, flows = load_cache(args.cache)
    sel = select_flows(flows)

    X, masks, labels, hasp, pstart0 = [], {r: [] for r in REGIONS}, [], [], []
    for f in sel:
        pr = prepare(f)
        if pr is None:
            continue
        X.append(torch.from_numpy(flow_to_image(pr["pkts"]).astype(np.float32) / 255.0))
        for r, m in region_masks(pr).items():
            masks[r].append(torch.from_numpy(m))
        labels.append(f["label"]); hasp.append(bool(pr["payload"].any())); pstart0.append(pr["pstarts"][0])
    X = torch.stack(X).to(device)
    labels, hasp = np.array(labels), np.array(hasp)
    pred, _ = predict(model, X)
    s = torch.cat([saliency(model, X[i:i + 256], pred[i:i + 256]) for i in range(0, len(X), 256)])
    tot = s.sum((1, 2, 3)) + 1e-12
    mass, share = {}, {}
    for r in REGIONS:
        mk = torch.stack(masks[r]).to(device).float()
        mass[r] = ((s * mk).sum((1, 2, 3)) / tot).cpu().numpy()
        share[r] = (mk.sum((1, 2, 3)) / mk[0].numel()).cpu().numpy()

    rows = []
    print(f"{'class':<22}{'region':<13}{'mass share':>11}{'pixel share':>13}{'enrichment':>12}")
    classes = [c for c in ATTACK_CLASSES if (labels == c).any()]
    for c in classes:
        m = labels == c
        for r in REGIONS:
            ms, ps = float(mass[r][m].mean()), float(share[r][m].mean())
            enr = ms / ps if ps > 0 else float("nan")
            rows.append([c, r, int(m.sum()), ms, float(mass[r][m].std()), ps, enr])
            print(f"{c:<22}{r:<13}{ms:>11.3f}{ps:>13.3f}{enr:>12.2f}")
    with open(out / f"saliency_region_mass_{args.tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["class", "region", "n_flows", "mass_mean", "mass_sd", "pixel_share_mean", "enrichment"]); w.writerows(rows)

    fig, ax = plt.subplots(figsize=(9, 4.4))
    bottom = np.zeros(len(classes))
    for r, col in zip(REGIONS, ["#4c72b0", "#d62728", "#c9d1d9", "#55a868", "#dd8452"]):
        v = np.array([[x[3] for x in rows if x[0] == c and x[1] == r][0] for c in classes])
        ax.bar([SHORT[c] for c in classes], v, bottom=bottom, color=col, label=r + (" (editable by the attack)" if r == "ch0_payload" else ""))
        bottom += v
    ax.set_ylabel("share of saliency mass"); ax.set_ylim(0, 1.02); ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax.set_title("Where the classifier looks, by region of the traffic image")
    fig.tight_layout(); fig.savefig(out / f"fig_saliency_regions_{args.tag}.png", dpi=200); plt.close(fig)

    sb = (s.sum((1, 2)).cpu().numpy()); sb = sb / (sb.sum(1, keepdims=True) + 1e-12)
    fig, ax = plt.subplots(figsize=(9, 4))
    for c in classes:
        m = labels == c
        ax.plot(np.arange(B), sb[m].mean(0), label=f"{SHORT[c]} (n={int(m.sum())})")
    ax.set_xlabel("byte offset inside the packet (0 = first byte of the IP header)"); ax.set_ylabel("mean saliency share")
    ax.set_title("Saliency by byte position (summed over packets and channels)"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out / f"fig_saliency_by_byte_{args.tag}.png", dpi=200); plt.close(fig)

    if Path(args.official).exists():
        o = np.load(args.official, allow_pickle=True)
        oe = sorted({int(k.split("_")[0]) for k in o.files if k.split("_")[0].isdigit()})
        olab = o[f"{oe[0]}_label"].astype(str)
        if len(olab) == int(hasp.sum()) and np.array_equal(olab, labels[hasp]):
            from scipy.stats import spearmanr
            from sklearn.metrics import roc_auc_score
            emax = max(oe)
            op0 = o[f"{oe[0]}_clean_pred"].astype(int); oa = o[f"{emax}_adv_pred"].astype(int)
            idx = {"DDoS-UDP_Flood": 1, "DoS-TCP_Flood": 2, "Recon-PortScan": 3}
            li = np.array([idx.get(l, -1) for l in olab])
            ok = (li >= 0) & (op0 == li)
            flip = ok & (oa != op0)
            pm = mass["ch0_payload"][hasp]
            print(f"\nDoes payload saliency predict who flips at eps {emax}? (official run, {int(ok.sum())} flows correct before the attack)")
            if flip[ok].sum() >= 3 and (~flip[ok]).sum() >= 3:
                print(f"  AUROC of payload saliency share for 'flipped': {roc_auc_score(flip[ok], pm[ok]):.3f}   Spearman rho {spearmanr(pm[ok], flip[ok].astype(float))[0]:.3f}")
            else:
                print("  too few flipped flows for an AUROC")
            print("  (assumes class index 1 DDoS, 2 DoS-TCP, 3 Recon, as in the baseline report; change idx if your model differs)")
        else:
            print("\n[warning] could not align with the official per-sample file; skipped the flip analysis.")
    print(f"\nwrote figures and CSV to {out}/")


if __name__ == "__main__":
    main()
