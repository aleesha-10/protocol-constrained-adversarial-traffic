"""Flow audit from the flow cache only: no GPU, no model, no PCAPs.

  python scripts/flow_audit.py
  python scripts/flow_audit.py --cache data/flows_cache.pkl --out results/audit

Answers two open questions in the report:
  1. Why did 282 of 600 sampled flows produce no attack result (149 of 150 DoS-TCP flows)?
     -> counts, per class, flows with NO editable payload pixel and the reason.
  2. Where do the 0.36% invalid checksums come from?
     -> verifies every stored packet BEFORE any edit. If the same share is already invalid, the failures come
        from the capture (e.g. checksum offload), not from the attack.
Writes flow_audit.json, class_summary.csv and flows_table.csv to --out.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from audit_utils import (ATTACK_CLASSES, MAX_FLOWS_PER_CLASS, P, B, load_cache, prepare, select_flows)
from src.attacks.protocol_utils import verify_checksums


def reason_no_payload(flow, pr):
    """Why a prepared flow has no editable pixel."""
    L = [len(p) for p in pr["packets"]]
    ps = pr["pstarts"]
    has_payload = [L[i] > ps[i] for i in range(len(L))]
    if not any(has_payload):
        return "no packet carries any payload bytes"
    first16 = [i for i in range(min(len(L), P)) if has_payload[i]]
    if not first16:
        return f"payload only after packet {P}"
    return f"payload starts after byte {B} in every one of the first {P} packets"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default="results/audit")
    ap.add_argument("--all-stored", action="store_true", help="audit every stored flow instead of the sampled ones")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sig, flows = load_cache(args.cache)
    print(f"cache signature: {sig}")
    sel = ([f for f in flows if f['label'] in ATTACK_CLASSES] if args.all_stored else select_flows(flows))
    print(f"auditing {len(sel)} flows ({'all stored' if args.all_stored else f'{MAX_FLOWS_PER_CLASS} sampled per class, same as the attack run'})\n")

    rows, per_class = [], {}
    pre_valid_by_flow = {}
    details = Counter()
    for idx, f in enumerate(sel):
        c = f["label"]
        d = per_class.setdefault(c, dict(selected=0, no_ip_packets=0, with_payload=0, no_payload=0, reasons=Counter(),
                                         editable=[], n_packets=[], pkt_total=0, pkt_invalid=0, pkt_invalid_payload_flows=0))
        d["selected"] += 1
        pr = prepare(f)
        if pr is None:
            d["no_ip_packets"] += 1
            rows.append([idx, c, len(f["pkts"]), 0, False, "no IPv4 packet"])
            continue
        n_edit = int(pr["payload"].sum())
        has = n_edit > 0
        reason = "" if has else reason_no_payload(f, pr)
        d["with_payload" if has else "no_payload"] += 1
        if has:
            d["editable"].append(n_edit)
        else:
            d["reasons"][reason] += 1
        d["n_packets"].append(len(pr["packets"]))
        ok_flags = []
        for p in pr["packets"]:
            ok, det = verify_checksums(bytes(p))
            ok_flags.append(ok)
            details[(c, "valid" if ok else "INVALID", det)] += 1
        d["pkt_total"] += len(ok_flags)
        d["pkt_invalid"] += len(ok_flags) - sum(ok_flags)
        if has:
            pre_valid_by_flow[idx] = float(np.mean(ok_flags))
        rows.append([idx, c, len(pr["packets"]), n_edit, has, reason])

    # ---- console + files ----
    print(f"{'class':<22}{'sampled':>8}{'with payload':>14}{'no payload':>12}   reasons (flows without editable pixels)")
    summ = {}
    for c in ATTACK_CLASSES:
        if c not in per_class:
            continue
        d = per_class[c]
        ed = np.array(d["editable"]) if d["editable"] else np.array([0])
        print(f"{c:<22}{d['selected']:>8}{d['with_payload']:>14}{d['no_payload']+d['no_ip_packets']:>12}   "
              + "; ".join(f"{k}: {v}" for k, v in d["reasons"].items()))
        summ[c] = dict(sampled=d["selected"], with_payload=d["with_payload"], no_payload=d["no_payload"],
                       no_ip_packets=d["no_ip_packets"], reasons=dict(d["reasons"]),
                       editable_pixels_mean=float(ed.mean()), editable_pixels_median=float(np.median(ed)),
                       editable_pixels_min=int(ed.min()), editable_pixels_max=int(ed.max()),
                       packets_checked=d["pkt_total"], packets_invalid_before_any_edit=d["pkt_invalid"])
    tot_with = sum(d["with_payload"] for d in per_class.values())
    tot = sum(d["selected"] for d in per_class.values())
    print(f"\nflows with an editable payload pixel: {tot_with} of {tot}  (the attack run reported 318 of 600)")

    pk, inv = sum(d["pkt_total"] for d in per_class.values()), sum(d["pkt_invalid"] for d in per_class.values())
    mean_pre = float(np.mean(list(pre_valid_by_flow.values()))) if pre_valid_by_flow else float("nan")
    print("\nChecksum audit BEFORE any edit (every stored packet, all sampled flows):")
    print(f"  packets checked: {pk:,} | invalid: {inv:,} ({100*inv/max(pk,1):.3f}%)")
    print(f"  mean per-flow valid share on flows that have editable payload: {mean_pre:.4f}   "
          f"(the attack run reported checksum validity 0.9964 at every epsilon)")
    print("  by class and reason:")
    for (c, s, det), n in sorted(details.items()):
        if s == "INVALID" or det not in ("tcp", "udp"):
            print(f"    {c:<22}{s:<8}{det:<26}{n:>9,}")
    if abs(mean_pre - 0.9964) < 0.002:
        print("  -> the pre-edit validity matches the attack's: the failures are already in the captured packets, not caused by the edits.")
    else:
        print("  -> the pre-edit validity differs from the attack's 0.9964: the edits (or the attack's packet set) matter; inspect the reasons above.")

    json.dump(dict(signature=[str(x) for x in sig], classes=summ, packets_checked=pk, packets_invalid_before_edit=inv,
                   mean_pre_edit_valid_share_with_payload=mean_pre,
                   checksum_details={f"{c}|{s}|{det}": n for (c, s, det), n in details.items()}),
              open(out / "flow_audit.json", "w"), indent=2)
    with open(out / "class_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "sampled", "with_payload", "no_payload", "editable_median", "editable_min", "editable_max"])
        for c, s in summ.items():
            w.writerow([c, s["sampled"], s["with_payload"], s["no_payload"] + s["no_ip_packets"],
                        s["editable_pixels_median"], s["editable_pixels_min"], s["editable_pixels_max"]])
    with open(out / "flows_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_index", "class", "packets_stored", "editable_pixels", "has_payload", "reason_if_none"])
        w.writerows(rows)
    print(f"\nwrote {out}/flow_audit.json, class_summary.csv, flows_table.csv")


if __name__ == "__main__":
    main()
