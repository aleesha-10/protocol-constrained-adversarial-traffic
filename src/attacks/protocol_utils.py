# protocol_util.py

"""Protocol-validity utilities for raw packets.


Pure-Python, byte-level, no dependency on the parsing library.
Assumes packets start at an Ethernet header (auto-detected, VLAN-aware)
or directly at the IPv4 header. Nothing except checksum fields is ever
modified by the repair function.
"""
import struct


def find_ip_offset(pkt: bytes) -> int:
    """Offset of the IPv4 header. Packets are stored as raw IPv4 datagrams
    (Ethernet stripped), so the common case is offset 0. Ethernet/VLAN
    detection is kept only for robustness with full frames."""
    if len(pkt) >= 20 and (pkt[0] >> 4) == 4 and (pkt[0] & 0x0F) >= 5:
        return 0
    if len(pkt) >= 14:
        et = struct.unpack("!H", pkt[12:14])[0]
        if et == 0x0800:
            off = 14
        elif et == 0x8100 and len(pkt) >= 18 and \
                struct.unpack("!H", pkt[16:18])[0] == 0x0800:
            off = 18
        else:
            return -1
        return off if (len(pkt) > off and pkt[off] >> 4 == 4) else -1
    return -1


def parse_ip(pkt: bytes, ip_off: int):
    ihl = (pkt[ip_off] & 0x0F) * 4
    total_len = struct.unpack("!H", pkt[ip_off + 2:ip_off + 4])[0]
    proto = pkt[ip_off + 9]
    return ihl, total_len, proto, pkt[ip_off + 12:ip_off + 16], pkt[ip_off + 16:ip_off + 20]


def _internet_checksum(data: bytes) -> int:
    if len(data) % 2:
        data += b"\x00"
    s = sum(struct.unpack("!%dH" % (len(data) // 2), data))
    s = (s & 0xFFFF) + (s >> 16)
    s = (s & 0xFFFF) + (s >> 16)
    return (~s) & 0xFFFF


def repair_transport_checksum(pkt: bytes) -> bytes:
    """Recompute the TCP/UDP checksum after payload edits.

    IP header checksum is untouched: it covers only the header, and payload
    edits never touch the header. Packet length is never changed, so IP
    total length and TCP seq/ack stay valid (sequence integrity).
    """
    pkt = bytearray(pkt)
    ip_off = find_ip_offset(bytes(pkt))
    if ip_off < 0:
        return bytes(pkt)
    ihl, total_len, proto, src, dst = parse_ip(pkt, ip_off)
    t_off, t_end = ip_off + ihl, ip_off + total_len
    min_hdr = 20 if proto == 6 else 8
    if proto not in (6, 17) or t_end > len(pkt) or t_end - t_off < min_hdr:
        return bytes(pkt)
    csum_rel = 16 if proto == 6 else 6
    csum_off = t_off + csum_rel
    pkt[csum_off:csum_off + 2] = b"\x00\x00"
    seg = bytes(pkt[t_off:t_end])
    pseudo = bytes(src) + bytes(dst) + struct.pack("!BBH", 0, proto, len(seg))
    pkt[csum_off:csum_off + 2] = struct.pack("!H", _internet_checksum(pseudo + seg))
    return bytes(pkt)


def verify_checksums(pkt: bytes):
    """Recompute checksums from stored fields. Returns (ok: bool, detail: str)."""
    ip_off = find_ip_offset(pkt)
    if ip_off < 0:
        return True, "non-ipv4"
    ihl, total_len, proto, src, dst = parse_ip(pkt, ip_off)
    if _internet_checksum(bytes(pkt[ip_off:ip_off + ihl])) != 0:
        return False, "ip-header-checksum"
    if proto not in (6, 17):
        return True, "no-transport-checksum"
    t_off, t_end = ip_off + ihl, ip_off + total_len
    if t_end > len(pkt):
        return True, "truncated-skipped"
    csum_rel = 16 if proto == 6 else 6
    stored = struct.unpack("!H", pkt[t_off + csum_rel:t_off + csum_rel + 2])[0]
    if proto == 17 and stored == 0:
        return True, "udp-zero-checksum"
    seg = bytearray(pkt[t_off:t_end])
    seg[csum_rel:csum_rel + 2] = b"\x00\x00"
    pseudo = bytes(src) + bytes(dst) + struct.pack("!BBH", 0, proto, len(seg))
    ok = _internet_checksum(pseudo + bytes(seg)) == stored
    return ok, "tcp" if proto == 6 else "udp"


def checksum_field_offsets(pkt: bytes) -> set:
    """Absolute offsets of IP + transport checksum fields (allowed to change)."""
    offs = set()
    ip_off = find_ip_offset(pkt)
    if ip_off < 0:
        return offs
    ihl, total_len, proto, _, _ = parse_ip(pkt, ip_off)
    offs.update(range(ip_off + 10, ip_off + 12))
    if proto in (6, 17) and ip_off + total_len <= len(pkt):
        rel = 16 if proto == 6 else 6
        t_off = ip_off + ihl
        offs.update(range(t_off + rel, t_off + rel + 2))
    return offs