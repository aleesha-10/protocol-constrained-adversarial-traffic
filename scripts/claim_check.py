"""Audit: do the numbers quoted in the report match the stored results?

  python scripts/claim_check.py
  python scripts/claim_check.py --baseline-json results/metrics/eval_baseline_summary.json
  python scripts/claim_check.py --dump computed.json       # write what the data says for every claim
  python scripts/claim_check.py --claims my_values.json    # check a JSON of {claim_id: quoted_value} instead

Every claim below is a number quoted in the Phase 3 report. The script recomputes it from the stored
per-sample file and prints PASS, FAIL or SKIPPED. When your report changes (new run, new numbers), edit the
quoted values in CLAIMS or pass --claims.

Some quantities can be defined two ways (over known-class flows only, or over all flows with a result).
A claim passes if either definition matches, and the table shows which one did.
"""
import argparse
import json
from pathlib import Path

import numpy as np

NAMES = ["Benign", "DDoS-UDP_Flood", "DoS-TCP_Flood", "Recon-PortScan"]
KNOWN = NAMES[1:]
UNSEEN = "DictionaryBruteForce"

# id, description, quoted value, decimals quoted with
CLAIMS = [
    ("n_rows", "flows with a result", 318, 0),
    ("n_DDoS", "DDoS-UDP flows with a result", 119, 0),
    ("n_DoS", "DoS-TCP flows with a result", 1, 0),
    ("n_Recon", "Recon-PortScan flows with a result", 94, 0),
    ("n_Brute", "DictionaryBruteForce flows with a result", 104, 0),
    ("n_known", "known-class flows with a result", 214, 0),
    ("correct_total", "known-class flows correct before the attack", 180, 0),
    ("correct_DDoS", "DDoS-UDP flows correct before the attack", 94, 0),
    ("correct_Recon", "Recon flows correct before the attack", 86, 0),
    ("acc_clean", "known-class accuracy, clean", 0.8411, 4),
    ("acc_1", "known-class accuracy, eps 1", 0.8411, 4),
    ("acc_4", "known-class accuracy, eps 4", 0.8364, 4),
    ("acc_16", "known-class accuracy, eps 16", 0.8037, 4),
    ("acc_32", "known-class accuracy, eps 32", 0.7336, 4),
    ("conf_clean", "mean confidence, clean", 0.8713, 4),
    ("conf_1", "mean confidence, eps 1", 0.8698, 4),
    ("conf_4", "mean confidence, eps 4", 0.8651, 4),
    ("conf_16", "mean confidence, eps 16", 0.8541, 4),
    ("conf_32", "mean confidence, eps 32", 0.7960, 4),
    ("bytes_1", "bytes changed per flow, eps 1", 138.2, 1),
    ("bytes_4", "bytes changed per flow, eps 4", 142.6, 1),
    ("bytes_16", "bytes changed per flow, eps 16", 145.0, 1),
    ("bytes_32", "bytes changed per flow, eps 32", 144.6, 1),
    ("checksum_valid", "checksum validity", 0.9964, 4),
    ("hdr_violations", "header violations (total over all eps)", 0, 0),
    ("realization_max", "realization max difference", 0.0, 4),
    ("DDoS_flips_16", "DDoS flips at eps 16", 2, 0),
    ("DDoS_flips_32", "DDoS flips at eps 32", 10, 0),
    ("DDoS_tobenign_32", "DDoS flips to Benign at eps 32", 3, 0),
    ("Recon_flips_4", "Recon flips at eps 4", 1, 0),
    ("Recon_flips_16", "Recon flips at eps 16", 6, 0),
    ("Recon_flips_32", "Recon flips at eps 32", 13, 0),
    ("Recon_tobenign_32", "Recon flips to Benign at eps 32", 12, 0),
    ("DDoS_rate_32", "DDoS flip rate at eps 32", 0.106, 3),
    ("Recon_rate_32", "Recon flip rate at eps 32", 0.151, 3),
    ("DDoS_ci_lo_32", "DDoS Wilson lower bound at eps 32", 0.059, 3),
    ("DDoS_ci_hi_32", "DDoS Wilson upper bound at eps 32", 0.185, 3),
    ("Recon_ci_lo_32", "Recon Wilson lower bound at eps 32", 0.091, 3),
    ("Recon_ci_hi_32", "Recon Wilson upper bound at eps 32", 0.242, 3),
    ("Brute_changed_1", "unseen class: predictions changed at eps 1", 3, 0),
    ("Brute_changed_4", "unseen class: predictions changed at eps 4", 6, 0),
    ("Brute_changed_16", "unseen class: predictions changed at eps 16", 18, 0),
    ("Brute_changed_32", "unseen class: predictions changed at eps 32", 28, 0),
    ("naive_evasion_DDoS_1", "script's 'evasion rate' for DDoS at eps 1", 0.210, 3),
]

# numbers from the baseline report, searched for anywhere in eval_baseline_summary.json
BASELINE_CLAIMS = [
    ("overall accuracy", 0.9669, 4), ("ECE", 0.0244, 4),
    ("Benign precision", 0.9673, 4), ("Benign recall", 0.9744, 4), ("Benign F1", 0.9709, 4),
    ("DDoS precision", 0.9881, 4), ("DDoS recall", 0.9749, 4), ("DDoS F1", 0.9815, 4),
    ("DoS-TCP precision", 0.9000, 4), ("DoS-TCP recall", 0.6667, 4), ("DoS-TCP F1", 0.7660, 4),
    ("Recon precision", 0.9553, 4), ("Recon recall", 0.9567, 4), ("Recon F1", 0.9560, 4),
    ("Benign support", 2736, 0), ("DDoS support", 1195, 0), ("DoS-TCP support", 27, 0), ("Recon support", 2147, 0),
    ("held-out -> Benign", 2156, 0), ("held-out -> Recon", 750, 0), ("held-out -> DDoS", 117, 0), ("held-out -> DoS-TCP", 15, 0),
    ("held-out mean max-softmax", 0.7731, 4), ("known mean max-softmax", 0.9426, 4), ("held-out mean entropy", 0.5210, 4),
]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / den, (c + h) / den


def load(path):
    d = np.load(path, allow_pickle=True)
    eps = sorted({int(k.split("_")[0]) for k in d.files if k.split("_")[0].isdigit()})
    lab = d[f"{eps[0]}_label"].astype(str)
    li = np.array([NAMES.index(l) if l in KNOWN else -1 for l in lab])
    D = dict(eps=eps, lab=lab, li=li, p0=d[f"{eps[0]}_clean_pred"].astype(int), c0=d[f"{eps[0]}_clean_conf"].astype(float))
    for p in ["clean"] + [str(e) for e in eps]:
        for k in ("adv_pred", "adv_conf", "n_changed", "valid_rate", "hdr_violations", "realization_diff"):
            if f"{p}_{k}" in d.files:
                D[(p, k)] = d[f"{p}_{k}"].astype(float)
    return D


def compute(D):
    """Return {claim_id: {definition_label: value}}; claims whose epsilon is missing are left out."""
    lab, li, p0 = D["lab"], D["li"], D["p0"]
    known = li >= 0
    correct0 = known & (p0 == li)
    R = {"n_rows": {"rows": len(lab)}, "n_known": {"known": int(known.sum())},
         "correct_total": {"known": int(correct0.sum())}}
    for tag, name in (("DDoS", "DDoS-UDP_Flood"), ("DoS", "DoS-TCP_Flood"), ("Recon", "Recon-PortScan"), ("Brute", UNSEEN)):
        R[f"n_{tag}"] = {"rows": int((lab == name).sum())}
    R["correct_DDoS"] = {"known": int((correct0 & (lab == "DDoS-UDP_Flood")).sum())}
    R["correct_Recon"] = {"known": int((correct0 & (lab == "Recon-PortScan")).sum())}
    hv, rd, vr = [], [], []
    for p in ["clean"] + [str(e) for e in D["eps"]]:
        if (p, "adv_pred") not in D:
            continue
        ap = D[(p, "adv_pred")].astype(int)
        ac = D[(p, "adv_conf")]
        R[f"acc_{p}"] = {"known": float((ap[known] == li[known]).mean())}
        R[f"conf_{p}"] = {"known flows": float(ac[known].mean()), "all flows": float(ac.mean())}
        if p != "clean":
            nc = D[(p, "n_changed")]
            R[f"bytes_{p}"] = {"all flows": float(nc.mean()), "known flows": float(nc[known].mean())}
            hv.append(D[(p, "hdr_violations")].sum()); rd.append(D[(p, "realization_diff")].max())
            vr.append(D[(p, "valid_rate")])
            flip = correct0 & (ap != p0)
            for tag, name in (("DDoS", "DDoS-UDP_Flood"), ("Recon", "Recon-PortScan")):
                m = lab == name
                n_ok = int(correct0[m].sum()); k = int(flip[m].sum())
                lo, hi = wilson(k, n_ok)
                R[f"{tag}_flips_{p}"] = {"count": k}
                R[f"{tag}_tobenign_{p}"] = {"count": int((flip & (ap == 0) & m).sum())}
                R[f"{tag}_rate_{p}"] = {"rate": k / n_ok if n_ok else float("nan")}
                R[f"{tag}_ci_lo_{p}"] = {"lo": lo}
                R[f"{tag}_ci_hi_{p}"] = {"hi": hi}
            mb = lab == UNSEEN
            R[f"Brute_changed_{p}"] = {"count": int((ap[mb] != p0[mb]).sum())}
            m = lab == "DDoS-UDP_Flood"
            R[f"naive_evasion_DDoS_{p}"] = {"error rate": float((ap[m] != li[m]).mean())}
    if hv:
        R["hdr_violations"] = {"sum": float(sum(hv))}
        R["realization_max"] = {"max": float(max(rd))}
        v = np.stack(vr)
        R["checksum_valid"] = {"all flows": float(v.mean()), "known flows": float(v[:, known].mean())}
    return R


def flatten(o, path=""):
    out = []
    if isinstance(o, dict):
        for k, v in o.items():
            out += flatten(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out += flatten(v, f"{path}[{i}]")
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        out.append((path, float(o)))
    return out


def close(a, b, dec):
    return abs(a - b) <= 0.5 * 10 ** (-dec) + 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/metrics/adversarial_per_sample_OFFICIAL.npz")
    ap.add_argument("--baseline-json", default="results/metrics/eval_baseline_summary.json")
    ap.add_argument("--claims", default=None, help="JSON of {claim_id: quoted_value} replacing the quoted values")
    ap.add_argument("--dump", default=None, help="write the computed value of every claim to this JSON file")
    args = ap.parse_args()

    R = compute(load(args.npz))
    quoted = {}
    if args.claims:
        quoted = json.load(open(args.claims))
    bad = skipped = ok = 0
    print(f"{'claim':<24}{'quoted':>10}  {'status':<8} computed")
    dump = {}
    for cid, desc, q, dec in CLAIMS:
        q = quoted.get(cid, q)
        cand = R.get(cid)
        if cand is None:
            print(f"{cid:<24}{q:>10}  SKIPPED  (not in this file: {desc})")
            skipped += 1
            continue
        dump[cid] = list(cand.values())[0]
        hit = [k for k, v in cand.items() if v == v and close(v, q, dec)]
        shown = ", ".join(f"{k}={v:.{max(dec, 0) + 1}f}" if isinstance(v, float) else f"{k}={v}" for k, v in cand.items())
        if hit:
            ok += 1
            print(f"{cid:<24}{q:>10}  PASS     {shown}")
        else:
            bad += 1
            print(f"{cid:<24}{q:>10}  FAIL     {shown}   <- {desc}")
    if Path(args.baseline_json).exists():
        leaves = flatten(json.load(open(args.baseline_json)))
        print(f"\nBaseline numbers searched in {args.baseline_json} ({len(leaves)} numeric values found):")
        for name, q, dec in BASELINE_CLAIMS:
            hits = [p for p, v in leaves if close(v, q, dec)]
            if hits:
                ok += 1
                print(f"  PASS  {name:<28}{q:>9}  at {hits[0]}" + (f" (+{len(hits)-1} more)" if len(hits) > 1 else ""))
            else:
                bad += 1
                print(f"  FAIL  {name:<28}{q:>9}  not found in the file (it may live in eval_baseline.npz or be computed elsewhere)")
    else:
        print(f"\n(No baseline file at {args.baseline_json}; baseline numbers not checked.)")
    if args.dump:
        json.dump(dump, open(args.dump, "w"), indent=1)
        print(f"\nComputed values written to {args.dump}")
    print(f"\n{ok} pass, {bad} fail, {skipped} skipped")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
