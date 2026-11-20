# Save as scripts/length_timing_attack.py
import csv
import struct
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import torch
import torch.nn.functional as F
from scapy.layers.inet import IP, UDP

from attack_controls import load_model, predict
from src.data.dataset import PCAPImageDataset
from src.data.flows import parse_pcap_flows
from src.encoding.flow_to_image import B, P, flow_to_image

CLASSES = ["Benign", "DDoS-UDP_Flood", "DoS-TCP_Flood", "Recon-PortScan"]
ATTACK_CLASSES = ["DDoS-UDP_Flood", "DoS-TCP_Flood", "Recon-PortScan"]


def checksum(data: bytes) -> int:
    if len(data) % 2:
        data += b"\x00"
    s = sum(struct.unpack(f"!{len(data)//2}H", data))
    s = (s >> 16) + (s & 0xFFFF)
    s += s >> 16
    return (~s) & 0xFFFF


def pad_udp_ip_packet(raw_ip: bytes, pad_bytes: int) -> bytes:
    """Appends pad_bytes of zeros to a UDP/IPv4 packet and recomputes IP + UDP lengths & checksums."""
    if pad_bytes <= 0 or len(raw_ip) < 28 or (raw_ip[0] >> 4) != 4 or raw_ip[9] != 17:
        return raw_ip
    ihl = (raw_ip[0] & 0x0F) * 4
    if len(raw_ip) < ihl + 8:
        return raw_ip

    max_pad = max(0, 1500 - len(raw_ip))
    pad = min(pad_bytes, max_pad)
    if pad == 0:
        return raw_ip

    buf = bytearray(raw_ip) + (b"\x00" * pad)
    new_ip_len = len(buf)
    struct.pack_into("!H", buf, 2, new_ip_len)
    struct.pack_into("!H", buf, 10, 0)
    struct.pack_into("!H", buf, 10, checksum(bytes(buf[:ihl])))

    new_udp_len = new_ip_len - ihl
    struct.pack_into("!H", buf, ihl + 4, new_udp_len)
    struct.pack_into("!H", buf, ihl + 6, 0)  # valid optional UDP checksum over IPv4
    return bytes(buf)


def verify_packet_validity(packets) -> tuple[bool, int]:
    """Checks Scapy parsing and ensures no header invariant violations."""
    violations = 0
    for _, raw in packets:
        try:
            pkt = IP(raw)
            if pkt.len != len(raw):
                violations += 1
            if UDP in pkt and pkt[UDP].len != len(raw) - pkt[IP].ihl * 4:
                violations += 1
        except Exception:
            violations += 1
    return (violations == 0), violations


def load_heldout_test_flows(data_dir="data/raw", max_per_class=150):
    """Loads held-out test split flows (last 15% chronologically) matching Step 3 split."""
    selected = []
    for cls_name in CLASSES:
        cls_folder = Path(data_dir) / cls_name
        cap = 150000 if cls_name == "DoS-TCP_Flood" else 50000
        class_flows = []
        for pcap_file in sorted(cls_folder.glob("*.pcap")):
            valid_flows, _ = parse_pcap_flows(pcap_file, max_packets=cap, min_packets=4)
            for fkey, pkts in valid_flows.items():
                class_flows.append((pkts[0][0], cls_name, pkts[:P]))
        class_flows.sort(key=lambda x: x[0])
        class_flows = class_flows[:2000]
        i_val = int(0.85 * len(class_flows))
        test_flows = class_flows[i_val:][:max_per_class]
        selected.extend(test_flows)
    return selected


def perturb_flow_greedy(model, device, packets, true_idx, max_pad_bytes, max_delay_sec, steps=15):
    """
    Coordinate/random-search black-box & gradient-guided packet-space attack:
    optimizes per-packet UDP zero-padding (0..max_pad_bytes) and non-negative delays (0..max_delay_sec),
    re-encoding via flow_to_image at every candidate so every step is 100% wire-realizable.
    """
    n = len(packets)
    best_pads = [0] * n
    best_delays = [0.0] * n

    def build_pkts(pads, delays):
        out = []
        cum_delay = 0.0
        for i, (t, raw) in enumerate(packets):
            cum_delay += delays[i]
            new_raw = pad_udp_ip_packet(raw, pads[i]) if pads[i] > 0 else raw
            out.append((t + cum_delay, new_raw))
        return out

    def score(pkts):
        img = flow_to_image(pkts).astype(np.float32) / 255.0
        x = torch.from_numpy(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0]
            pred = int(probs.argmax().item())
            loss = -F.cross_entropy(logits, torch.tensor([true_idx], device=device)).item()
        return pred, float(probs.max().item()), loss

    best_pkts = build_pkts(best_pads, best_delays)
    best_pred, best_conf, best_loss = score(best_pkts)
    if best_pred != true_idx:
        return best_pkts, best_pred, best_conf, sum(best_pads), sum(best_delays)

    rng = np.random.default_rng(42)
    for _ in range(steps):
        cand_pads = list(best_pads)
        cand_delays = list(best_delays)
        idx = int(rng.integers(0, n))
        if max_pad_bytes > 0:
            cand_pads[idx] = int(rng.integers(0, max_pad_bytes + 1))
        if max_delay_sec > 0.0 and idx > 0:
            cand_delays[idx] = float(rng.uniform(0.0, max_delay_sec))

        cand_pkts = build_pkts(cand_pads, cand_delays)
        pred, conf, loss = score(cand_pkts)
        if pred != true_idx or loss < best_loss:
            best_pads, best_delays = cand_pads, cand_delays
            best_pkts, best_pred, best_conf, best_loss = cand_pkts, pred, conf, loss
            if best_pred != true_idx:
                break

    return best_pkts, best_pred, best_conf, sum(best_pads), sum(best_delays)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, c2i = load_model("results/metrics/resnet18_pcap_model.pth", device)
    flows = load_heldout_test_flows(max_per_class=150)
    print(f"Loaded {len(flows)} held-out test flows on {device}", flush=True)

    # Arms: (arm_name, max_pad_bytes, max_delay_sec)
    arms = [
        ("pad_only_256B", 256, 0.0),
        ("pad_only_1024B", 1024, 0.0),
        ("timing_only_100ms", 0, 0.10),
        ("timing_only_1000ms", 0, 1.00),
        ("joint_pad1024B_delay1000ms", 1024, 1.00),
    ]

    out_dir = Path("results/length_timing")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for arm_name, max_pad, max_delay in arms:
        print(f"\nRunning arm: {arm_name}...", flush=True)
        stats = {cls: {"correct0": 0, "flips": 0, "to_benign": 0, "pad_bytes": [], "delay_s": [], "valid": 0, "violations": 0} for cls in ATTACK_CLASSES}

        for _, cls_name, pkts in flows:
            if cls_name not in ATTACK_CLASSES:
                continue
            true_idx = c2i[cls_name]
            clean_img = torch.from_numpy(flow_to_image(pkts).astype(np.float32) / 255.0).unsqueeze(0).to(device)
            p0, _ = predict(model, clean_img)
            if int(p0.item()) != true_idx:
                continue

            stats[cls_name]["correct0"] += 1
            adv_pkts, adv_pred, _, tot_pad, tot_delay = perturb_flow_greedy(
                model, device, pkts, true_idx, max_pad, max_delay, steps=20
            )
            ok, viol = verify_packet_validity(adv_pkts)
            stats[cls_name]["valid"] += int(ok)
            stats[cls_name]["violations"] += viol
            stats[cls_name]["pad_bytes"].append(tot_pad)
            stats[cls_name]["delay_s"].append(tot_delay)

            if adv_pred != true_idx:
                stats[cls_name]["flips"] += 1
                if adv_pred == c2i["Benign"]:
                    stats[cls_name]["to_benign"] += 1

        for cls_name in ATTACK_CLASSES:
            s = stats[cls_name]
            n_ok = s["correct0"]
            rate = s["flips"] / n_ok if n_ok > 0 else 0.0
            mean_pad = float(np.mean(s["pad_bytes"])) if s["pad_bytes"] else 0.0
            mean_del = float(np.mean(s["delay_s"])) if s["delay_s"] else 0.0
            valid_rate = s["valid"] / n_ok if n_ok > 0 else 1.0
            print(
                f"  {cls_name:<18} | flips: {s['flips']}/{n_ok} ({rate:.3f}) "
                f"| to_Benign: {s['to_benign']} | mean_pad={mean_pad:.1f}B | mean_delay={mean_del:.3f}s "
                f"| valid={valid_rate:.3f} | violations={s['violations']}"
            )
            summary_rows.append({
                "arm": arm_name,
                "class": cls_name,
                "correct_clean": n_ok,
                "flips": s["flips"],
                "flip_rate": round(rate, 4),
                "to_benign": s["to_benign"],
                "mean_pad_bytes": round(mean_pad, 2),
                "mean_delay_sec": round(mean_del, 4),
                "valid_rate": round(valid_rate, 4),
                "hdr_violations": s["violations"],
            })

    csv_path = out_dir / "length_timing_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    main()