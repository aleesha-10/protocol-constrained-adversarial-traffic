"""Shared helpers. They reproduce adversarial_attack.py's flow selection and preprocessing EXACTLY, so results
from the new scripts line up flow-for-flow with adversarial_per_sample.npz.

The cache file is a tuple (signature, flows): pickle.load(...) returns TWO objects. That is why code that
iterated over it as a list of flows failed.

If adversarial_attack.py can be imported (torch and timm installed) its constants are used; otherwise the copies
below are used. If you change a constant in adversarial_attack.py, change it here too.
"""
import pickle
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.attacks.protocol_utils import find_ip_offset            # noqa: E402  (pure Python)
from src.encoding.flow_to_image import P, B, flow_to_image       # noqa: E402  (numpy only)

try:                                                              # use the real constants when possible
    import adversarial_attack as AA
    ATTACK_CLASSES, SEED = list(AA.ATTACK_CLASSES), AA.SEED
    MAX_FLOWS_PER_CLASS, PKT_STRIDE = AA.MAX_FLOWS_PER_CLASS, AA.PKT_STRIDE
    EPS_BYTES, STEPS, ALPHA_BYTES = list(AA.EPS_BYTES), AA.STEPS, AA.ALPHA_BYTES
    CACHE_PATH, MODEL_PATH = Path(AA.CACHE_PATH), AA.MODEL_PATH
except Exception:                                                 # torch/timm missing: fall back to copies
    ATTACK_CLASSES = ["DDoS-UDP_Flood", "DoS-TCP_Flood", "Recon-PortScan", "DictionaryBruteForce"]
    SEED, MAX_FLOWS_PER_CLASS, PKT_STRIDE = 42, 150, 100000
    EPS_BYTES, STEPS, ALPHA_BYTES = [1, 4, 16, 32], 25, 2
    CACHE_PATH, MODEL_PATH = Path("data/flows_cache.pkl"), "results/metrics/resnet18_pcap_model.pth"


def load_cache(path=None):
    sig, flows = pickle.load(open(path or CACHE_PATH, "rb"))
    return sig, flows


def select_flows(flows, cap=None, seed=None):
    """Same sampling as adversarial_attack.main(): shuffle each class with Random(SEED), take the first `cap`."""
    by_class = {}
    for f in flows:
        if f["label"] in ATTACK_CLASSES:
            by_class.setdefault(f["label"], []).append(f)
    rng = random.Random(SEED if seed is None else seed)
    cap = MAX_FLOWS_PER_CLASS if cap is None else cap
    sel = []
    for cname in ATTACK_CLASSES:
        fs = by_class.get(cname, [])
        rng.shuffle(fs)
        sel.extend(fs[:cap])
    return sel


def build_payload_byte_map(packets, pstarts, H, W):
    """Same as adversarial_attack.build_payload_byte_map: pixel -> (row*PKT_STRIDE + byte offset) for payload bytes."""
    m = np.full((H, W), -1, np.int64)
    for r in range(min(len(packets), H)):
        s, n = pstarts[r], min(len(packets[r]), W)
        if n > s:
            m[r, s:n] = np.arange(s, n) + r * PKT_STRIDE
    return m


def prepare(flow):
    """Same preprocessing as run_flow(): drop packets without an IPv4 header, build the editable-pixel map.
    Returns None if the flow has no usable packet."""
    pairs = [(t, p, s) for (t, p), s in zip(flow["pkts"], flow["pstarts"]) if find_ip_offset(p) >= 0]
    if not pairs:
        return None
    pkts = [(t, p) for t, p, _ in pairs]
    packets = [p for _, p, _ in pairs]
    pstarts = [s for _, _, s in pairs]
    bmap = build_payload_byte_map(packets, pstarts, P, B)
    any_byte = np.zeros((P, B), bool)                     # pixels that correspond to a real byte of a real packet
    for r in range(min(len(packets), P)):
        any_byte[r, :min(len(packets[r]), B)] = True
    return dict(pkts=pkts, packets=packets, pstarts=pstarts, bmap=bmap, payload=(bmap >= 0), any_byte=any_byte)


def official_flow_list(sel):
    """The flows that produced a row in adversarial_per_sample.npz, in the same order."""
    out = []
    for f in sel:
        pr = prepare(f)
        if pr is not None and pr["payload"].any():
            out.append((f, pr))
    return out
