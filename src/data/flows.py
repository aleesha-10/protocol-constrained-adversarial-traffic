# Save as src/data/flows.py
import socket
from collections import defaultdict
from pathlib import Path
import dpkt

def parse_pcap_flows(pcap_file, max_packets=50000, min_packets=4):
    """Unified flow extractor: keyed by (protocol, sorted_endpoints), drops flows < min_packets."""
    flows = defaultdict(list)
    dropped_short = 0
    pkt_count = 0

    with open(pcap_file, "rb") as f:
        try:
            reader = dpkt.pcap.Reader(f)
        except ValueError:
            f.seek(0)
            reader = dpkt.pcapng.Reader(f)

        for ts, buf in reader:
            try:
                eth = dpkt.ethernet.Ethernet(buf)
                if not isinstance(eth.data, dpkt.ip.IP):
                    continue
                ip = eth.data
                proto = getattr(ip, "p", 0)
                src_ip = socket.inet_ntoa(getattr(ip, "src"))
                dst_ip = socket.inet_ntoa(getattr(ip, "dst"))

                if isinstance(ip.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                    sport = getattr(ip.data, "sport", 0)
                    dport = getattr(ip.data, "dport", 0)
                else:
                    sport, dport = 0, 0

                ep1, ep2 = (src_ip, sport), (dst_ip, dport)
                endpoints = (ep1, ep2) if ep1 <= ep2 else (ep2, ep1)
                flow_key = (proto,) + endpoints

                flows[flow_key].append((float(ts), bytes(ip)))
                pkt_count += 1
                if max_packets is not None and pkt_count >= max_packets:
                    break
            except Exception:
                continue

    valid_flows = {}
    for k, pkts in flows.items():
        if len(pkts) < min_packets:
            dropped_short += 1
        else:
            valid_flows[k] = pkts

    return valid_flows, dropped_short