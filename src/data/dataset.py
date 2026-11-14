from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset
import numpy as np
from scapy.all import IP, TCP, UDP, PcapReader

from src.encoding.flow_to_image import B, P, flow_to_image


def _extract_windows(pcap_file, max_packets, max_windows_per_flow):
    flows = defaultdict(list)
    packet_count = 0

    with PcapReader(str(pcap_file)) as reader:
        for packet in reader:
            if IP not in packet:
                continue

            ip = packet[IP]
            transport = packet[TCP] if TCP in packet else (packet[UDP] if UDP in packet else None)
            source_port, destination_port = (
                (transport.sport, transport.dport) if transport is not None else (0, 0)
            )
            source = (ip.src, source_port)
            destination = (ip.dst, destination_port)
            endpoints = (source, destination) if source <= destination else (destination, source)
            flow_key = (ip.proto,) + endpoints

            flows[flow_key].append((float(packet.time), bytes(ip)))
            packet_count += 1
            if packet_count >= max_packets:
                break

    windows = []
    for packets in flows.values():
        if len(packets) < 4:
            continue
        for start in range(0, len(packets), P):
            if start >= max_windows_per_flow * P:
                break
            windows.append(flow_to_image(packets[start:start + P]))

    if windows:
        return np.stack(windows)
    return np.empty((0, 3, P, B), dtype=np.uint8)

class PCAPImageDataset(Dataset):
    def __init__(self, root_dir, max_packets_per_file=50000, max_windows_per_file=20, exclude_classes=None):
        """
        Scans data/raw/<class_name>/*.pcap, extracts flow windows, 
        and flattens them into individual image samples for training.
        
        exclude_classes: list of folder names to skip (e.g., hold-out classes for evaluation)
        """
        self.root_dir = Path(root_dir)
        self.samples = []  # Stores tuples of (window_tensor, label_idx)
        exclude_classes = exclude_classes or []
        
        # Discover all valid training class folders (excluding hold-outs)
        all_dirs = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.classes = [cls for cls in all_dirs if cls not in exclude_classes]
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print(f"Active Training Classes & Indices: {self.class_to_idx}")
        if exclude_classes:
            print(f"Excluded Hold-out Classes: {exclude_classes}")

        for cls_name in self.classes:
            cls_folder = self.root_dir / cls_name
            label_idx = self.class_to_idx[cls_name]
            
            for pcap_file in cls_folder.glob("*.pcap"):
                X = _extract_windows(
                    pcap_file,
                    max_packets=max_packets_per_file,
                    max_windows_per_flow=max_windows_per_file,
                )
                if len(X) > 0:
                    for i in range(len(X)):
                        self.samples.append((X[i], label_idx))
                        
        print(f"Loaded total of {len(self.samples)} window samples across {len(self.classes)} active classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_np, label = self.samples[idx]
        # Convert uint8 numpy array [3, 16, 64] to float32 PyTorch tensor scaled to [0.0, 1.0]
        x_tensor = torch.from_numpy(x_np.astype(np.float32) / 255.0)
        y_tensor = torch.tensor(label, dtype=torch.long)
        return x_tensor, y_tensor