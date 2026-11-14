# adversarial_attack.py
"""Protocol-constrained adversarial perturbation + evasion evaluation.

Attacker model (this variant):
  * may modify ONLY application-payload bytes (offset >= pstart) of the first
    64 bytes of the first 16 packets - exactly the pixels the encoder exposes
  * may not add/drop packets or change lengths -> channel 1 (length) and
    channel 2 (timing) are frozen; this attack tests payload-content evasion
  * must leave valid IP/TCP/UDP checksums (packets stay valid on the wire)

Pipeline per flow, per epsilon:
  clean encode -> PGD on channel-0 payload pixels -> realize rounded byte
  deltas -> repair transport checksums -> verify validity + header
  immutability -> re-encode -> measure evasion.

Outputs: results/metrics/adversarial_results.json + adversarial_per_sample.npz
"""
import json
import pickle
import random
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import torch
import torch.nn.functional as F
import timm

from src.attacks.protocol_utils import (find_ip_offset, repair_transport_checksum,
                                        verify_checksums, checksum_field_offsets)

DATA_DIR = "data/raw"
MODEL_PATH = "results/metrics/resnet18_pcap_model.pth"
EPS_BYTES = [1, 4, 16, 32]        # Linf budget per payload byte (byte units)
ALPHA_BYTES = 2                 # PGD step size (byte units)
STEPS = 25
TARGETED = False
TARGET_CLASS = "Benign"         # used only when TARGETED = True
ATTACK_CLASSES = ["DDoS-UDP_Flood", "DoS-TCP_Flood", "Recon-PortScan",
                  "DictionaryBruteForce"]      # holdout runs in pseudo-label mode
MAX_FLOWS_PER_CLASS = 150
MAX_FLOWS_STORED_PER_CLASS = 600
MAX_PKTS_PER_FLOW = 48
SMOKE_TEST = False              # True = tiny fast end-to-end check
SEED = 42
PKT_STRIDE = 100000
OUT_DIR = Path("results/metrics")
CACHE_PATH = Path("data/flows_cache.pkl")
STALE_PACKETS_LIMIT = 50000
POST_CAP_PACKETS  = 300000   # hard stop: max packets scanned after flow cap

def encode_packets(pkts):
    """pkts: list of (timestamp, ip_bytes), time-ordered - exactly what
    flow_to_image() expects. Returns [3, P, B] float in [0,1]."""
    from src.encoding.flow_to_image import flow_to_image
    img = flow_to_image(pkts)                  # uint8 (3, P, B)
    return torch.from_numpy(img.astype(np.float32) / 255.0)


def load_flows(data_dir):
    """Streaming loader -> [{"pkts": [(t, ip_bytes), ...], "pstarts": [...],
    "label": str}, ...]. Bytes start at the IPv4 header (Ethernet stripped).

    Early-stop rules per class (whichever fires first):
      1. cap reached AND every stored flow is full
      2. STALE_PACKETS_LIMIT consecutive packets added nothing
      3. POST_CAP_PACKETS scanned since the cap was reached (deterministic
         backstop: single stray packets keep resetting rule 2 on this data)
    """
    from scapy.all import PcapReader, IP, TCP, UDP, Raw
    flows, per_class, full_cnt, stale, since_cap = {}, {}, {}, {}, {}
    for class_dir in sorted(p for p in Path(data_dir).iterdir() if p.is_dir()):
        if class_dir.name not in ATTACK_CLASSES:
            continue
        cname = class_dir.name
        for pcap in sorted(class_dir.glob("*.pcap")):
            if per_class.get(cname, 0) >= MAX_FLOWS_STORED_PER_CLASS and \
                    (full_cnt.get(cname, 0) >= per_class[cname]
                     or since_cap.get(cname, 0) >= POST_CAP_PACKETS):
                print(f"  skipping {cname}/{pcap.name}: class storage already full",
                      flush=True)
                continue
            print(f"  parsing {cname}/{pcap.name} ...", flush=True)
            n = 0
            with PcapReader(str(pcap)) as pr:
                for pkt in pr:
                    n += 1
                    if per_class.get(cname, 0) >= MAX_FLOWS_STORED_PER_CLASS:
                        since_cap[cname] = since_cap.get(cname, 0) + 1
                    if n % 100000 == 0:
                        print(f"    {n:,} packets | {per_class.get(cname, 0)} flows "
                              f"stored, {full_cnt.get(cname, 0)} full", flush=True)
                        if per_class.get(cname, 0) >= MAX_FLOWS_STORED_PER_CLASS and \
                                (full_cnt.get(cname, 0) >= per_class[cname]
                                 or stale.get(cname, 0) >= STALE_PACKETS_LIMIT
                                 or since_cap.get(cname, 0) >= POST_CAP_PACKETS):
                            print("    early break: nothing more can be stored",
                                  flush=True)
                            break
                    useful = False
                    if IP in pkt:
                        if TCP in pkt:
                            sport, dport = int(pkt[TCP].sport), int(pkt[TCP].dport)
                        elif UDP in pkt:
                            sport, dport = int(pkt[UDP].sport), int(pkt[UDP].dport)
                        else:
                            sport = dport = None
                        if sport is not None:
                            ipb = bytes(pkt[IP])
                            pstart = len(ipb) - len(bytes(pkt[Raw])) if Raw in pkt else len(ipb)
                            ip = pkt[IP]
                            key = (cname, tuple(sorted([(ip.src, sport),
                                                        (ip.dst, dport)])))
                            f = flows.get(key)
                            if f is None:
                                if per_class.get(cname, 0) < MAX_FLOWS_STORED_PER_CLASS:
                                    f = {"pkts": [], "pstarts": [], "label": cname}
                                    flows[key] = f
                                    per_class[cname] = per_class.get(cname, 0) + 1
                                    useful = True
                            elif len(f["pkts"]) < MAX_PKTS_PER_FLOW:
                                useful = True
                            if useful and len(f["pkts"]) < MAX_PKTS_PER_FLOW:
                                f["pkts"].append((float(pkt.time), ipb))
                                f["pstarts"].append(pstart)
                                if len(f["pkts"]) >= MAX_PKTS_PER_FLOW:
                                    full_cnt[cname] = full_cnt.get(cname, 0) + 1
                    stale[cname] = 0 if useful else stale.get(cname, 0) + 1
            print(f"    scanned {n:,} packets | {per_class.get(cname, 0)} flows "
                  f"stored ({full_cnt.get(cname, 0)} full)", flush=True)
    return [f for f in flows.values() if len(f["pkts"]) >= 2]


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else 0.0


def _max(xs):
    return float(np.max(xs)) if len(xs) else 0.0


def build_payload_byte_map(packets, pstarts, H, W):
    """HxW int map: pixel -> (pkt_idx * PKT_STRIDE + byte_off) for MUTABLE
    payload bytes only; -1 for headers, padding, unmapped pixels. Offsets are
    within the IP datagram; row r = packet r."""
    m = np.full((H, W), -1, np.int64)
    for r in range(min(len(packets), H)):
        s, n = pstarts[r], min(len(packets[r]), W)
        if n > s:
            m[r, s:n] = np.arange(s, n) + r * PKT_STRIDE
    return m


def masked_pgd(model, x0, y, mask, eps_pix, alpha_pix, n_steps, targeted):
    """PGD applied to channel 0 only (packet-byte pixels). Channels 1/2 are
    functions of length/timing, which in-place byte edits cannot change, so
    their gradients are irrelevant and only g[0] updates dp."""
    dp = torch.zeros_like(x0[:1])                 # [1, P, B]
    for _ in range(n_steps):
        xa = x0.clone()
        xa[0:1] = (x0[0:1] + dp).clamp(0.0, 1.0)
        xa = xa.detach().requires_grad_(True)
        logits = model(xa.unsqueeze(0))
        ce = F.cross_entropy(logits, y)           # y is already shape (1,)
        loss = -ce if targeted else ce
        g = torch.autograd.grad(loss, xa)[0]      # [3, P, B]
        dp = dp + alpha_pix * g[0:1].sign()       # channel-0 gradient only
        dp = torch.where(mask, dp, torch.zeros_like(dp)).clamp(-eps_pix, eps_pix)
    return dp[0]                                  # [P, B] pixel units


def realize(packets, pstarts, dp, bmap):
    """Round dp*255 into byte deltas and write them into payload bytes only."""
    new_packets = [bytearray(p) for p in packets]
    valid = bmap >= 0
    if dp is None or not valid.any():
        return new_packets, 0, 0
    pkt_idx = bmap[valid] // PKT_STRIDE
    off = bmap[valid] % PKT_STRIDE
    db = np.round(dp.cpu().numpy()[valid] * 255.0).astype(int)
    keep = db != 0
    changed = 0
    linf = 0
    for pi, o, d in zip(pkt_idx[keep], off[keep], db[keep]):
        if o < pstarts[pi]:                       # belt and braces: never touch headers
            continue
        orig = new_packets[pi][o]
        nb = int(np.clip(orig + d, 0, 255))
        if nb != orig:
            new_packets[pi][o] = nb
            changed += 1
            linf = max(linf, abs(nb - orig))
    return new_packets, changed, linf


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    eps_list, n_steps, cap = (([2], 3, 5) if SMOKE_TEST
                              else (EPS_BYTES, STEPS, MAX_FLOWS_PER_CLASS))

    ckpt = torch.load(MODEL_PATH, map_location=device)
    state_dict, class_to_idx = ckpt["state_dict"], ckpt["class_to_idx"]
    num_classes = state_dict["fc.weight"].shape[0]
    target_y = torch.tensor([class_to_idx[TARGET_CLASS]], device=device)

    cache_sig = ("v3-earlybreak", tuple(ATTACK_CLASSES), MAX_PKTS_PER_FLOW,
                 MAX_FLOWS_STORED_PER_CLASS)
    
    flows = None
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "rb") as fh:
                sig, flows = pickle.load(fh)
            if sig != cache_sig:
                flows = None
        except Exception:
            flows = None
    if flows is None:
        print("Loading flows (this parses the PCAPs)...")
        flows = load_flows(DATA_DIR)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "wb") as fh:
            pickle.dump((cache_sig, flows), fh)
        print(f"Cached flows to {CACHE_PATH}")
    print(f"Loaded {len(flows)} flows")

    by_class = {}
    for f in flows:
        if f["label"] in ATTACK_CLASSES:
            by_class.setdefault(f["label"], []).append(f)
    rng = random.Random(SEED)
    sel = []
    for cname in ATTACK_CLASSES:
        fs = by_class.get(cname, [])
        rng.shuffle(fs)
        sel.extend(fs[:cap])
        print(f"  {cname}: attacking {min(len(fs), cap)} of {len(fs)} stored flows")
    if not sel:
        print("No flows selected - check class folder names under data/raw.")
        return

    model = timm.create_model(ckpt.get("model", "resnet18"), pretrained=False,
                              in_chans=ckpt.get("in_chans", 3), num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    def run_flow(flow, eps_bytes):
        pairs = [(t, p, s) for (t, p), s in zip(flow["pkts"], flow["pstarts"])
                 if find_ip_offset(p) >= 0]
        if not pairs:
            return None
        pkts = [(t, p) for t, p, _ in pairs]
        packets = [p for _, p, _ in pairs]
        pstarts = [s for _, _, s in pairs]
        x0 = encode_packets(pkts).to(device)                  # [3, P, B]
        bmap = build_payload_byte_map(packets, pstarts, x0.shape[1], x0.shape[2])
        mask = torch.from_numpy(bmap >= 0).unsqueeze(0).to(device)
        if not bool(mask.any()):
            return None
        with torch.no_grad():
            lg = model(x0.unsqueeze(0))
            clean_pred = int(lg.argmax(1))
            clean_conf = float(F.softmax(lg, 1).max())
        if TARGETED:
            y = target_y
        else:
            y = torch.tensor([class_to_idx.get(flow["label"], clean_pred)],
                             device=device)
        rec = {"label": flow["label"],
               "known": flow["label"] in class_to_idx,
               "clean_pred": clean_pred, "clean_conf": clean_conf}
        if eps_bytes is None:
            rec.update({"adv_pred": clean_pred, "adv_conf": clean_conf,
                        "n_changed": 0, "linf": 0, "valid_rate": 1.0,
                        "hdr_violations": 0, "realization_diff": 0.0})
            return rec

        eps_pix = eps_bytes / 255.0
        alpha_pix = min(ALPHA_BYTES, eps_bytes) / 255.0
        dp = masked_pgd(model, x0, y, mask, eps_pix, alpha_pix, n_steps, TARGETED)
        new_packets, n_changed, linf = realize(packets, pstarts, dp, bmap)

        violations = 0
        for i, (o, nw) in enumerate(zip(packets, new_packets)):
            if bytes(o) != bytes(nw):
                new_packets[i] = bytearray(repair_transport_checksum(bytes(nw)))
            allowed = checksum_field_offsets(bytes(new_packets[i]))
            for j in range(min(len(o), len(new_packets[i]))):
                if o[j] != new_packets[i][j] and j < pstarts[i] and j not in allowed:
                    violations += 1
                    break
        valid_cnt = sum(1 for p in new_packets if verify_checksums(bytes(p))[0])

        x_adv = encode_packets([(t, bytes(nb))
                                for (t, _), nb in zip(pkts, new_packets)]).to(device)
        realization_diff = float((x_adv[0] - (x0[0] + dp).clamp(0, 1)).abs().max())
        with torch.no_grad():
            lg = model(x_adv.unsqueeze(0))
            adv_pred = int(lg.argmax(1))
            adv_conf = float(F.softmax(lg, 1).max())
        rec.update({"adv_pred": adv_pred, "adv_conf": adv_conf,
                    "n_changed": n_changed, "linf": linf,
                    "valid_rate": valid_cnt / len(new_packets),
                    "hdr_violations": violations,
                    "realization_diff": realization_diff})
        return rec

    print(f"\nClean pass on {len(sel)} flows...", flush=True)
    all_recs = {"clean": [r for r in (run_flow(f, None) for f in sel) if r]}
    for e in eps_list:
        print(f"\n=== epsilon = {e} byte(s) ===", flush=True)
        recs = [r for r in (run_flow(f, e) for f in sel) if r]
        all_recs[str(e)] = recs
        known = [r for r in recs if r["known"]]
        ev = _mean([r["adv_pred"] != class_to_idx[r["label"]] for r in known])
        print(f"known-class evasion rate: {ev:.3f} | "
              f"mean conf {_mean([r['adv_conf'] for r in recs]):.3f} | "
              f"bytes changed/flow {_mean([r['n_changed'] for r in recs]):.1f} | "
              f"checksum validity {_mean([r['valid_rate'] for r in recs]):.4f} | "
              f"header violations {int(sum(r['hdr_violations'] for r in recs))} | "
              f"realization max diff {_max([r['realization_diff'] for r in recs]):.4f}")

    def summarize(recs):
        known = [r for r in recs if r["known"]]
        out = {"n": len(recs),
               "known_acc": _mean([r["adv_pred"] == class_to_idx[r["label"]]
                                   for r in known]),
               "mean_conf": _mean([r["adv_conf"] for r in recs]),
               "mean_bytes_changed": _mean([r["n_changed"] for r in recs]),
               "mean_valid_rate": _mean([r["valid_rate"] for r in recs]),
               "header_violations": int(sum(r["hdr_violations"] for r in recs)),
               "realization_max_diff": _max([r["realization_diff"] for r in recs])}
        if TARGETED and known:
            out["targeted_success"] = _mean([r["adv_pred"] == class_to_idx[TARGET_CLASS]
                                             for r in recs])
        return out

    summary = {"config": {"eps_bytes": eps_list, "alpha_bytes": ALPHA_BYTES,
                          "steps": n_steps, "targeted": TARGETED,
                          "target_class": TARGET_CLASS if TARGETED else None,
                          "attack_classes": ATTACK_CLASSES,
                          "max_flows_per_class": cap,
                          "smoke_test": SMOKE_TEST, "seed": SEED},
               "clean": summarize(all_recs["clean"]),
               "eps": {e: summarize(all_recs[e]) for e in all_recs if e != "clean"}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "adversarial_results.json").write_text(json.dumps(summary, indent=2))

    keys = ["label", "known", "clean_pred", "clean_conf", "adv_pred", "adv_conf",
            "n_changed", "linf", "valid_rate", "hdr_violations", "realization_diff"]
    arrs = {}
    for tag, recs in all_recs.items():
        for k in keys:
            vals = [r[k] for r in recs]
            arrs[f"{tag}_{k}"] = np.array(vals, dtype=object if k == "label" else float)
    np.savez(OUT_DIR / "adversarial_per_sample.npz", **arrs)
    print(f"\nSaved: {OUT_DIR/'adversarial_results.json'} + adversarial_per_sample.npz")


if __name__ == "__main__":
    main()