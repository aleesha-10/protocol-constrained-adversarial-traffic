"""Step 1 feasibility test: PCAP -> flows -> images.

Usage:
  python scripts/feasibility_check.py --pcap data/raw/Wednesday.pcap --max-packets 200000

Prints throughput and flow stats, saves sample flow images, and prints a
recommendation (Option A packet-level vs Option B CSV-grid).
"""
import argparse, sys, time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from scapy.all import PcapReader, IP, TCP, UDP

from src.encoding.flow_to_image import flow_to_image, P, B


def flow_key(pkt):
    ip = pkt[IP]
    l4 = pkt[TCP] if TCP in pkt else (pkt[UDP] if UDP in pkt else None)
    sp, dp = (l4.sport, l4.dport) if l4 is not None else (0, 0)
    a, b = (ip.src, sp), (ip.dst, dp)
    return (ip.proto,) + (a + b if a <= b else b + a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", required=True)
    ap.add_argument("--max-packets", type=int, default=200000)
    ap.add_argument("--min-flow-packets", type=int, default=4)
    ap.add_argument("--n-samples", type=int, default=12)
    ap.add_argument("--out", default="results/figures/feasibility")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    flows = defaultdict(list)
    n = 0
    t0 = time.time()
    with PcapReader(args.pcap) as reader:
        for pkt in reader:
            if IP not in pkt:
                continue
            flows[flow_key(pkt)].append((float(pkt.time), bytes(pkt[IP])))
            n += 1
            if n >= args.max_packets:
                break
    dt = time.time() - t0
    rate = n / dt if dt > 0 else 0

    usable = {k: v for k, v in flows.items() if len(v) >= args.min_flow_packets}
    print(f"Read {n} IPv4 packets in {dt:.1f}s ({rate:,.0f} pkt/s)")
    print(f"Flows found: {len(flows)} | with >= {args.min_flow_packets} packets: {len(usable)}")

    for i, (k, pkts) in enumerate(list(usable.items())[: args.n_samples]):
        img = flow_to_image(pkts)                      # (3, P, B)
        arr = np.transpose(img, (1, 2, 0))             # (P, B, 3)
        Image.fromarray(arr).resize((B * 6, P * 6), Image.NEAREST).save(out / f"flow_{i:02d}.png")
    print(f"Saved {min(args.n_samples, len(usable))} sample images to {out}/ (image shape = 3x{P}x{B})")

    # Rough extrapolation: full CICIDS2017 is ~ 50M+ packets per big day.
    full_est_min = (50_000_000 / rate / 60) if rate else float("inf")
    print(f"Est. time to parse a 50M-packet day: ~{full_est_min:.0f} min")
    print("\nRecommendation:")
    if rate > 5000 and len(usable) > 100:
        print(" -> Option A (packet-level images) looks feasible. Parse a SUBSET of flows per class.")
    else:
        print(" -> Parsing is slow or few usable flows found. Consider Option B (CSV feature grid) or a smaller subset.")


if __name__ == "__main__":
    main()
