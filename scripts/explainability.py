# Save as scripts/explainability.py
import csv
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from attack_controls import load_model
from src.data.dataset import PCAPImageDataset

CLASSES = ["Benign", "DDoS-UDP_Flood", "DoS-TCP_Flood", "Recon-PortScan"]


def compute_attributions(ckpt_path, tag):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, c2i = load_model(ckpt_path, device)
    test_ds = PCAPImageDataset("data/raw", split="test", exclude_classes=["DictionaryBruteForce"])
    loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    region_sums = {cls: {"ch0_ip_hdr": [], "ch0_l4_hdr": [], "ch0_payload": [], "ch1_length": [], "ch2_timing": []} for cls in CLASSES}

    for x, y, pmask in loader:
        x = x.to(device).requires_grad_(True)
        y = y.to(device)
        logits = model(x)
        target_logits = logits.gather(1, y.unsqueeze(1)).sum()
        model.zero_grad()
        target_logits.backward()

        # Gradient x Input saliency magnitude: [B, 3, 16, 64]
        sal = (x.grad * x).abs().detach().cpu().numpy()
        pm = pmask.numpy()  # [B, 16, 64]
        y_np = y.cpu().numpy()

        for i in range(len(y_np)):
            cls_name = CLASSES[y_np[i]]
            s = sal[i]
            tot = s.sum() + 1e-12

            ip_mass = s[0, :, :20].sum() / tot
            # L4 header = cols 20..63 where payload mask == 0
            non_ip_ch0 = s[0, :, 20:]
            pm_sub = pm[i, :, 20:]
            payload_mass = (non_ip_ch0 * pm_sub).sum() / tot
            l4_mass = (non_ip_ch0 * (1.0 - pm_sub)).sum() / tot
            len_mass = s[1].sum() / tot
            time_mass = s[2].sum() / tot

            region_sums[cls_name]["ch0_ip_hdr"].append(ip_mass)
            region_sums[cls_name]["ch0_l4_hdr"].append(l4_mass)
            region_sums[cls_name]["ch0_payload"].append(payload_mass)
            region_sums[cls_name]["ch1_length"].append(len_mass)
            region_sums[cls_name]["ch2_timing"].append(time_mass)

    out_dir = Path("results/explainability")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"\nInput Attribution Share (Gradient x Input) [{tag}]:")
    print(f"{'Class':<18} {'IP_Hdr':>8} {'L4_Hdr':>8} {'Payload':>8} {'Ch1_Len':>8} {'Ch2_Time':>9}")
    for cls_name in CLASSES:
        r = region_sums[cls_name]
        row = {
            "model": tag,
            "class": cls_name,
            "ch0_ip_hdr": round(float(np.mean(r["ch0_ip_hdr"])), 4),
            "ch0_l4_hdr": round(float(np.mean(r["ch0_l4_hdr"])), 4),
            "ch0_payload": round(float(np.mean(r["ch0_payload"])), 4),
            "ch1_length": round(float(np.mean(r["ch1_length"])), 4),
            "ch2_timing": round(float(np.mean(r["ch2_timing"])), 4),
        }
        rows.append(row)
        print(
            f"{cls_name:<18} {row['ch0_ip_hdr']:>8.3f} {row['ch0_l4_hdr']:>8.3f} "
            f"{row['ch0_payload']:>8.3f} {row['ch1_length']:>8.3f} {row['ch2_timing']:>9.3f}"
        )

    csv_path = out_dir / f"channel_attribution_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    compute_attributions("results/metrics/resnet18_pcap_model.pth", "standard")