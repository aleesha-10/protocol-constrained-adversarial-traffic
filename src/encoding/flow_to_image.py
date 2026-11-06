"""Flow -> image encoder.

Each flow becomes a 3-channel image of shape (3, P, B):
  channel 0: first B bytes of each of the first P packets (IPv4 header onward)
  channel 1: packet length per row (scaled 0-255, broadcast across the row)
  channel 2: log inter-arrival time per row (scaled 0-255, broadcast across the row)

Leakage fields (IP addresses, L4 ports, checksums) are zeroed in channel 0.

FIELD_MAP documents which image regions correspond to which protocol fields.
Phase 3 Step 5 builds the perturbation mask from it, so keep it accurate.
"""
import numpy as np

P = 16   # packets per flow image (rows)
B = 64   # bytes per packet (columns)

IP_SRC = slice(12, 16)
IP_DST = slice(16, 20)
IP_CHECKSUM = slice(10, 12)


def _zero_leakage(raw: bytes) -> bytearray:
    buf = bytearray(raw[:B].ljust(B, b"\x00"))
    if len(raw) < 20 or (raw[0] >> 4) != 4:
        return buf  # not IPv4, leave as is
    ihl = (raw[0] & 0x0F) * 4
    buf[IP_SRC] = b"\x00" * 4
    buf[IP_DST] = b"\x00" * 4
    buf[IP_CHECKSUM] = b"\x00" * 2
    proto = raw[9]
    if proto in (6, 17) and ihl + 4 <= B:      # TCP / UDP ports
        buf[ihl:ihl + 4] = b"\x00" * 4
    if proto == 6 and ihl + 18 <= B:           # TCP checksum
        buf[ihl + 16:ihl + 18] = b"\x00" * 2
    elif proto == 17 and ihl + 8 <= B:         # UDP checksum
        buf[ihl + 6:ihl + 8] = b"\x00" * 2
    return buf


def flow_to_image(packets):
    """packets: list of (timestamp: float, ip_bytes: bytes), time ordered.
    Returns uint8 array of shape (3, P, B)."""
    img = np.zeros((3, P, B), dtype=np.uint8)
    prev_t = None
    for i, (t, raw) in enumerate(packets[:P]):
        img[0, i, :] = np.frombuffer(bytes(_zero_leakage(raw)), dtype=np.uint8)
        img[1, i, :] = min(255, int(len(raw) / 1500 * 255))
        if prev_t is not None:
            iat = max(t - prev_t, 0.0)
            img[2, i, :] = min(255, int(np.log1p(iat * 1000) / np.log1p(10000) * 255))
        prev_t = t
    return img


# Documentation of what each region means (used later to build the mask).
FIELD_MAP = {
    "channel0_ip_header_fixed": "rows 0..P-1, cols 0..19 (IP header; addresses/checksum zeroed)",
    "channel0_l4_header": "cols ihl..ihl+19 (TCP/UDP header; ports/checksum zeroed)",
    "channel0_payload": "cols after L4 header (attacker-influenceable: padding / app data)",
    "channel1_length": "per-row packet length (attacker-controllable via padding)",
    "channel2_timing": "per-row inter-arrival time (attacker-controllable via delays)",
}
