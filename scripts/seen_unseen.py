"""A held-out estimate for the EXISTING model: which cached flows did it never see during training?

train.py trains on windows from the first 50,000 IP packets of each PCAP and never splits the data, so the
baseline report is a training-set fit. But the attack's flow cache scans further into each file. A flow whose first
four packets lie after that 50,000-packet prefix cannot have been in training. This script labels each cached
flow seen / unseen and compares clean accuracy, and the attack's flips, between the two groups.

  python scripts/seen_unseen.py --raw data/raw
Needs the PCAPs (to find the end of the prefix, about 15 s per file), the cache and the model.
Caveats: first window only (the first 16 packets), later flows of the same capture (same attacker devices),
and flows with 2-3 packets are never in training. It is an estimate, not a clean test split.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scapy.all import IP, PcapReader

from attack_controls import load_model, predict, wilson
from audit_utils import ATTACK_CLASSES, MODEL_PATH, flow_to_image, load_cache, official_flow_list, prepare, select_flows

TRAIN_PREFIX_IP_PACKETS = 50000        # PCAPImageDataset default max_packets_per_file


def prefix_window(pcap, n_ip=TRAIN_PREFIX_IP_PACKETS):
    first = last = None
    cnt = 0
    with PcapReader(str(pcap)) as r:
        for pkt in r:
            if IP not in pkt:
                continue
            t = float(pkt.time)
            first = t if first is None else first
            last = t
            cnt += 1
            if cnt >= n_ip:
                break
    return first, last, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--ckpt", default=MODEL_PATH)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--official", default="results/metrics/adversarial_per_sample_OFFICIAL.npz")
    ap.add_argument("--out", default="results/seen_unseen")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, c2i = load_model(args.ckpt, device)
    _, flows = load_cache(args.cache)

    windows = {}
    for c in ATTACK_CLASSES:
        if c not in c2i:
            continue
        files = sorted((Path(args.raw) / c).glob("*.pcap"))
        windows[c] = [prefix_window(f) for f in files]
        for f, (a, b, n) in zip(files, windows[c]):
            print(f"  {c}/{f.name}: training prefix = first {n:,} IP packets, t = {a:.3f} .. {b:.3f}")

    def seen(flow):
        if len(flow["pkts"]) < 4:
            return False
        t4 = flow["pkts"][3][0]
        return any(a <= t4 <= b for a, b, _ in windows.get(flow["label"], []))

    items = []
    for f in flows:
        if f["label"] in c2i and f["label"] != "Benign":
            pr = prepare(f)
            if pr is not None:
                items.append((f, pr))
    X = torch.stack([torch.from_numpy(flow_to_image(pr["pkts"]).astype(np.float32) / 255.0) for _, pr in items]).to(device)
    pred, _ = predict(model, X)
    pred = pred.cpu().numpy()
    labels = np.array([f["label"] for f, _ in items]); y = np.array([c2i[l] for l in labels])
    is_seen = np.array([seen(f) for f, _ in items])
    names = [n for n, _ in sorted(c2i.items(), key=lambda kv: kv[1])]

    rows, summ = [], {}
    print(f"\n{'class':<22}{'group':<8}{'flows':>7}{'clean acc (first window)':>26}")
    for c in ATTACK_CLASSES:
        if c not in c2i:
            continue
        for g, gm in (("seen", is_seen), ("unseen", ~is_seen)):
            m = (labels == c) & gm
            n = int(m.sum()); k = int((pred[m] == y[m]).sum())
            lo, hi = wilson(k, n)
            rows.append([c, g, n, k, k / n if n else float("nan"), lo, hi])
            summ.setdefault(c, {})[g] = dict(n=n, correct=k, acc=(k / n if n else None), ci=[lo, hi])
            print(f"{c:<22}{g:<8}{n:>7}   {k}/{n} = {(k / n if n else float('nan')):.3f}  (95% CI {lo:.3f}-{hi:.3f})")
    um = ~is_seen
    cm = np.zeros((len(names), len(names)), int)
    for t, p in zip(y[um], pred[um]):
        cm[t, p] += 1
    print("\nConfusion matrix on UNSEEN flows (rows = true class, columns = predicted):")
    print("  columns: " + ", ".join(names))
    for i, n in enumerate(names):
        if cm[i].sum():
            print(f"  {n:<22}{cm[i].tolist()}")

    if Path(args.official).exists():
        sel = select_flows(flows)
        off = official_flow_list(sel)
        o = np.load(args.official, allow_pickle=True)
        oe = sorted({int(k.split("_")[0]) for k in o.files if k.split("_")[0].isdigit()})
        olab = o[f"{oe[0]}_label"].astype(str)
        if len(off) == len(olab) and all(f["label"] == l for (f, _), l in zip(off, olab)):
            emax = max(oe)
            op0 = o[f"{oe[0]}_clean_pred"].astype(int); oa = o[f"{emax}_adv_pred"].astype(int)
            sflag = np.array([seen(f) for f, _ in off])
            idx = {"DDoS-UDP_Flood": 1, "DoS-TCP_Flood": 2, "Recon-PortScan": 3}
            print(f"\nOfficial attack run, eps {emax}: flips among flows correct before the attack, seen vs unseen")
            for c in ("DDoS-UDP_Flood", "Recon-PortScan"):
                for g, gm in (("seen", sflag), ("unseen", ~sflag)):
                    m = (olab == c) & gm
                    ok = m & (op0 == idx[c])
                    k = int((ok & (oa != op0)).sum()); n = int(ok.sum())
                    lo, hi = wilson(k, n)
                    print(f"  {c:<18}{g:<8}{k}/{n}" + (f" = {k/n:.3f} (CI {lo:.3f}-{hi:.3f})" if n else ""))
        else:
            print("\n[warning] could not align with the official per-sample file; skipped the flip comparison.")
    json.dump(summ, open(out / "seen_unseen.json", "w"), indent=2)
    with open(out / "seen_unseen.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["class", "group", "flows", "correct", "acc", "ci_lo", "ci_hi"]); w.writerows(rows)
    print(f"\nwrote {out}/seen_unseen.json and .csv")


if __name__ == "__main__":
    main()
