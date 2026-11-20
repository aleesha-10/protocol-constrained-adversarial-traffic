# Save as scripts/train_adv.py
import csv
import json
import os
import random
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

from explainability import compute_attributions
from length_timing_attack import load_heldout_test_flows, perturb_flow_greedy, verify_packet_validity
from src.data.dataset import PCAPImageDataset
from src.encoding.flow_to_image import flow_to_image
from train import compute_ece, evaluate_loader, set_seed

CLASSES = ["Benign", "DDoS-UDP_Flood", "DoS-TCP_Flood", "Recon-PortScan"]
ATTACK_CLASSES = ["DDoS-UDP_Flood", "DoS-TCP_Flood", "Recon-PortScan"]


def make_protocol_adv_batch(model, x, y, pmask, eps=32.0 / 255.0, steps=5):
    """
    Generates protocol-aware adversarial training batch:
      1. 5-step PGD on Channel 0 valid payload pixels (pmask == 1), quantized to 8-bit grid.
      2. Random protocol-valid Channel 1 length padding (+0..0.40) and Channel 2 IAT delay (+0..0.35).
    """
    was_training = model.training
    model.eval()
    B_sz = x.size(0)
    mask3 = torch.zeros_like(x)
    mask3[:, 0] = pmask

    delta = torch.empty_like(x).uniform_(-eps, eps) * mask3
    alpha = (8.0 / 255.0)
    for _ in range(steps):
        x_in = torch.round(torch.clamp(x + delta, 0.0, 1.0) * 255.0) / 255.0
        x_in.requires_grad_(True)
        loss = F.cross_entropy(model(x_in), y)
        grad = torch.autograd.grad(loss, x_in)[0]
        delta = torch.clamp(delta + alpha * grad.sign() * mask3, -eps, eps)

    x_adv = torch.round(torch.clamp(x + delta, 0.0, 1.0) * 255.0) / 255.0

    # Random side-channel length padding (Ch 1) and delay jitter (Ch 2) on half the batch
    side_mask = (torch.rand(B_sz, 1, 1, device=x.device) < 0.5).float()
    pad_shift = torch.rand(B_sz, 16, 1, device=x.device) * 0.40 * side_mask
    time_shift = torch.rand(B_sz, 16, 1, device=x.device) * 0.35 * side_mask
    x_adv[:, 1] = torch.clamp(x_adv[:, 1] + pad_shift, 0.0, 1.0)
    x_adv[:, 2] = torch.clamp(x_adv[:, 2] + time_shift, 0.0, 1.0)

    if was_training:
        model.train()
    return x_adv.detach()


def train_adv_model(seed=42, out_path="results/metrics/resnet18_adv.pth"):
    set_seed(seed)
    data_dir = "data/raw"
    batch_size = 64
    epochs = 8
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Protocol-Aware Adversarial Training on {device} | seed: {seed}", flush=True)

    train_ds = PCAPImageDataset(data_dir, split="train", exclude_classes=["DictionaryBruteForce"])
    val_ds = PCAPImageDataset(data_dir, split="val", exclude_classes=["DictionaryBruteForce"])
    test_ds = PCAPImageDataset(data_dir, split="test", exclude_classes=["DictionaryBruteForce"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    num_classes = len(train_ds.classes)
    model = timm.create_model("resnet18", pretrained=False, in_chans=3, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(epochs):
        print(f"Starting Adv Epoch {epoch+1}/{epochs}...", flush=True)
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels, pmask in train_loader:
            inputs, labels, pmask = inputs.to(device), labels.to(device), pmask.to(device)
            adv_inputs = make_protocol_adv_batch(model, inputs, labels, pmask)

            optimizer.zero_grad()
            loss_clean = criterion(model(inputs), labels)
            loss_adv = criterion(model(adv_inputs), labels)
            loss = 0.5 * loss_clean + 0.5 * loss_adv
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            with torch.no_grad():
                correct += model(inputs).argmax(1).eq(labels).sum().item()

        val_acc, val_f1, val_ece, _, _, _ = evaluate_loader(model, val_loader, device)
        print(
            f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/total:.4f} "
            f"| Clean Train Acc: {correct/total*100:.2f}% | Val Acc: {val_acc*100:.2f}% "
            f"| Val Macro-F1: {val_f1:.4f} | Val ECE: {val_ece:.4f}",
            flush=True,
        )
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No model checkpoint was selected during training.")
    model.load_state_dict(best_state)
    test_acc, test_f1, test_ece, test_probs, test_y, test_pred = evaluate_loader(model, test_loader, device)
    print(f"\n=== ADV-TRAINED HELD-OUT TEST RESULTS ===")
    print(f"Test Accuracy: {test_acc:.4f} | Macro-F1: {test_f1:.4f} | ECE: {test_ece:.4f}")
    print(classification_report(test_y, test_pred, target_names=train_ds.classes, digits=4, zero_division=0))

    os.makedirs("results/metrics", exist_ok=True)
    torch.save(
        {
            "model": "resnet18",
            "in_chans": 3,
            "class_to_idx": train_ds.class_to_idx,
            "classes": train_ds.classes,
            "state_dict": best_state,
        },
        out_path,
    )

    with open("results/metrics/eval_adv_summary.json", "w") as f:
        json.dump(
            {
                "model": "resnet18_protocol_adv_trained",
                "test_accuracy": round(test_acc, 4),
                "macro_f1": round(test_f1, 4),
                "ECE": round(test_ece, 4),
            },
            f,
            indent=2,
        )

    # Evaluate defended model against Step 7 length/timing attacks
    print("\n=== EVALUATING DEFENDED MODEL AGAINST STEP 7 LENGTH/TIMING ATTACKS ===", flush=True)
    flows = load_heldout_test_flows(max_per_class=150)
    arms = [
        ("pad_only_1024B", 1024, 0.0),
        ("timing_only_1000ms", 0, 1.00),
        ("joint_pad1024B_delay1000ms", 1024, 1.00),
    ]
    c2i = train_ds.class_to_idx
    adv_rows = []
    model.eval()
    for arm_name, max_pad, max_delay in arms:
        stats = {cls: {"correct0": 0, "flips": 0, "to_benign": 0, "valid": 0, "violations": 0} for cls in ATTACK_CLASSES}
        for _, cls_name, pkts in flows:
            if cls_name not in ATTACK_CLASSES:
                continue
            true_idx = c2i[cls_name]
            clean_img = torch.from_numpy(flow_to_image(pkts).astype(np.float32) / 255.0).unsqueeze(0).to(device)
            with torch.no_grad():
                p0 = int(model(clean_img).argmax(1).item())
            if p0 != true_idx:
                continue
            stats[cls_name]["correct0"] += 1
            adv_pkts, adv_pred, _, _, _ = perturb_flow_greedy(model, device, pkts, true_idx, max_pad, max_delay, steps=20)
            ok, viol = verify_packet_validity(adv_pkts)
            stats[cls_name]["valid"] += int(ok)
            stats[cls_name]["violations"] += viol
            if adv_pred != true_idx:
                stats[cls_name]["flips"] += 1
                if adv_pred == c2i["Benign"]:
                    stats[cls_name]["to_benign"] += 1

        print(f"Arm: {arm_name}")
        for cls_name in ATTACK_CLASSES:
            s = stats[cls_name]
            n_ok = s["correct0"]
            rate = s["flips"] / n_ok if n_ok > 0 else 0.0
            print(f"  {cls_name:<18} | flips: {s['flips']}/{n_ok} ({rate:.3f}) | to_Benign: {s['to_benign']}")
            adv_rows.append({
                "model": "adv_trained",
                "arm": arm_name,
                "class": cls_name,
                "correct_clean": n_ok,
                "flips": s["flips"],
                "flip_rate": round(rate, 4),
                "to_benign": s["to_benign"],
                "valid_rate": round(s["valid"] / n_ok, 4) if n_ok > 0 else 1.0,
                "hdr_violations": s["violations"],
            })

    out_csv = Path("results/length_timing/length_timing_summary_adv_trained.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(adv_rows[0].keys()))
        w.writeheader()
        w.writerows(adv_rows)
    print(f"Saved {out_csv}")

    # Run Step 15 attribution on the defended model too
    compute_attributions(out_path, "adv_trained")


if __name__ == "__main__":
    train_adv_model()