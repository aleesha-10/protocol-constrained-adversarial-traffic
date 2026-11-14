"""Summarise who talks to whom in a PCAP, without full packet parsing (fast).

Why: CICIoT2023 attack PCAPs may mix attack packets with benign background traffic
from devices that were not being attacked. Before labelling a whole PCAP as one class,
look at the dominant IP / MAC pairs and decide which flows really are the attack.

Usage:
  python scripts/pcap_top_talkers.py --pcap data/raw/<file>.pcap --max-packets 500000
  python scripts/pcap_top_talkers.py --pcap data/raw/<file>.pcap --csv results/metrics/talkers_<name>.csv
Works with .pcap and .pcapng, Ethernet link type only.
"""
import argparse, csv, struct, time
from collections import Counter
from pathlib import Path

from scapy.utils import RawPcapReader, RawPcapNgReader


def open_reader(path):
    with open(path, "rb") as f:
        magic = f.read(4)
    return RawPcapNgReader(path) if magic == b"\x0a\x0d\x0d\x0a" else RawPcapReader(path)


def mac(b):
    return ":".join(f"{x:02x}" for x in b)


def ip4(b):
    return ".".join(str(x) for x in b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", required=True)
    ap.add_argument("--max-packets", type=int, default=500000)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--csv", default=None, help="optional path to save the IP-pair table")
    args = ap.parse_args()

    ip_pairs, mac_pairs, protos, dports = Counter(), Counter(), Counter(), Counter()
    n = non_ip = 0
    t0 = time.time()
    reader = open_reader(args.pcap)
    for pkt, _meta in reader:
        n += 1
        if len(pkt) >= 14:
            et = struct.unpack("!H", pkt[12:14])[0]
            mac_pairs[(mac(pkt[6:12]), mac(pkt[0:6]))] += 1
            off = 14
            if et == 0x8100 and len(pkt) >= 18:      # VLAN tag
                et = struct.unpack("!H", pkt[16:18])[0]
                off = 18
            if et == 0x0800 and len(pkt) >= off + 20:
                ihl = (pkt[off] & 0x0F) * 4
                proto = pkt[off + 9]
                ip_pairs[(ip4(pkt[off + 12:off + 16]), ip4(pkt[off + 16:off + 20]))] += 1
                protos[proto] += 1
                if proto in (6, 17) and len(pkt) >= off + ihl + 4:
                    dports[(proto, struct.unpack("!H", pkt[off + ihl + 2:off + ihl + 4])[0])] += 1
            else:
                non_ip += 1
        if n >= args.max_packets:
            break
    dt = time.time() - t0
    reader.close()

    total_ip = sum(ip_pairs.values())
    print(f"Read {n:,} packets in {dt:.1f}s | IPv4: {total_ip:,} | non-IPv4 (ARP, IPv6, etc.): {non_ip:,}")
    names = {1: "ICMP", 6: "TCP", 17: "UDP"}
    print("\nProtocols:", ", ".join(f"{names.get(p, p)}={c:,}" for p, c in protos.most_common()))
    print(f"\nTop {args.top} IP pairs (src -> dst):")
    cum = 0
    for (s, d), c in ip_pairs.most_common(args.top):
        cum += c
        print(f"  {s:>15} -> {d:<15} {c:>10,}  ({100*c/max(total_ip,1):5.1f}%)")
    print(f"  top-{args.top} pairs cover {100*cum/max(total_ip,1):.1f}% of IPv4 packets; {len(ip_pairs):,} distinct pairs in total")
    print(f"\nTop {args.top} MAC pairs (src -> dst):")
    for (s, d), c in mac_pairs.most_common(args.top):
        print(f"  {s} -> {d}  {c:>10,}")
    print(f"\nTop {args.top} destination ports (proto, port):")
    for (p, port), c in dports.most_common(args.top):
        print(f"  {names.get(p, p)}/{port:<6} {c:>10,}")

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["src_ip", "dst_ip", "packets"])
            for (s, d), c in ip_pairs.most_common():
                w.writerow([s, d, c])
        print(f"\nSaved IP-pair table to {args.csv}")

    print("\nHow to read this: if a few IP pairs dominate, those are likely the attack flows. "
          "If traffic is spread over many pairs, the file mixes background traffic, so do not label every flow as attack.")


if __name__ == "__main__":
    main()