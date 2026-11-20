# # train.py

# import os
# import sys
# from pathlib import Path

# # 1. Add project root to sys.path FIRST before any local imports
# root_dir = Path(__file__).resolve().parent.parent
# if str(root_dir) not in sys.path:
#     sys.path.insert(0, str(root_dir))

# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import timm

# # 2. Now import safely after path is registered
# from src.data.dataset import PCAPImageDataset

# def set_seed(seed=42):
#     """Sets seeds for complete reproducibility across runs."""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def train_model():
#     # Fix random seed for reproducibility (as promised in README)
#     set_seed(42)

#     # Hyperparameters & Setup
#     data_dir = "data/raw"
#     batch_size = 32
#     epochs = 5
#     lr = 1e-3
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # Dataset & DataLoader (Excluding hold-out class from training)
#     print("Loading dataset from PCAPs...")
#     dataset = PCAPImageDataset(root_dir=data_dir, exclude_classes=["DictionaryBruteForce"])
#     if len(dataset) == 0:
#         print("Error: No samples found! Check your data/raw/ folders and ensure .pcap files are present.")
#         return

#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
#     num_classes = len(dataset.classes)
#     print(f"Total samples: {len(dataset)} | Number of classes: {num_classes}")
#     print(f"Class mapping: {dataset.class_to_idx}")

#     # Model Initialization
#     model = timm.create_model('resnet18', pretrained=False, in_chans=3, num_classes=num_classes)
#     model.to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)

#     # Training Loop
#     print("\nStarting training loop...")
#     model.train()
#     for epoch in range(epochs):
#         running_loss = 0.0
#         correct = 0
#         total = 0

#         for batch_idx, (inputs, labels) in enumerate(dataloader):
#             inputs, labels = inputs.to(device), labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item() * inputs.size(0)
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()

#         epoch_loss = running_loss / total
#         epoch_acc = correct / total
#         print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc * 100:.2f}%")

#     # Save Trained Model Weights along with Metadata
#     os.makedirs("results/metrics", exist_ok=True)
#     model_path = "results/metrics/resnet18_pcap_model.pth"
    
#     checkpoint = {
#         "model": "resnet18",
#         "in_chans": 3,
#         "class_to_idx": dataset.class_to_idx,
#         "classes": dataset.classes,
#         "state_dict": model.state_dict(),
#     }
    
#     torch.save(checkpoint, model_path)
#     print(f"\nTraining complete! Model and metadata saved successfully to {model_path}")

# if __name__ == "__main__":
#     train_model()
# Save as scripts/train.py
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

from src.data.dataset import PCAPImageDataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_ece(probs, y_true, n_bins=15):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


@torch.no_grad()
def evaluate_loader(model, loader, device):
    model.eval()
    all_probs, all_y = [], []
    for inputs, labels, _ in loader:
        inputs = inputs.to(device)
        probs = F.softmax(model(inputs), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_y.append(labels.numpy())
    probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_y, axis=0)
    y_pred = probs.argmax(axis=1)
    acc = float((y_pred == y_true).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    ece = compute_ece(probs, y_true)
    return acc, macro_f1, ece, probs, y_true, y_pred


def train_model(seed=42, out_path="results/metrics/resnet18_pcap_model.pth"):
    set_seed(seed)
    data_dir = "data/raw"
    batch_size = 64
    epochs = 8
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | seed: {seed}")

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
        print(f"Starting Epoch {epoch+1}/{epochs}...", flush=True)
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            correct += outputs.argmax(1).eq(labels).sum().item()

        val_acc, val_f1, val_ece, _, _, _ = evaluate_loader(model, val_loader, device)
        print(
            f"Epoch [{epoch+1}/{epochs}] | Train Loss: {running_loss/total:.4f} "
            f"| Train Acc: {correct/total*100:.2f}% | Val Acc: {val_acc*100:.2f}% "
            f"| Val Macro-F1: {val_f1:.4f} | Val ECE: {val_ece:.4f}"
        )
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No validation checkpoint was produced; check that training completed.")
    model.load_state_dict(best_state)
    test_acc, test_f1, test_ece, test_probs, test_y, test_pred = evaluate_loader(model, test_loader, device)
    print(f"\n=== HELD-OUT TEST RESULTS (Best Val Checkpoint) ===")
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

    # Evaluate held-out unseen class (DictionaryBruteForce)
    unseen_ds = PCAPImageDataset(
        data_dir,
        split="all",
        exclude_classes=["Benign", "DDoS-UDP_Flood", "DoS-TCP_Flood", "Recon-PortScan"],
    )
    unseen_loader = DataLoader(unseen_ds, batch_size=128, shuffle=False)
    _, _, _, unseen_probs, _, unseen_pred = evaluate_loader(model, unseen_loader, device)
    holdout_conf = unseen_probs.max(axis=1)
    holdout_entropy = -(unseen_probs * np.log(np.clip(unseen_probs, 1e-12, 1.0))).sum(axis=1)
    pred_dist = np.bincount(unseen_pred, minlength=num_classes)

    np.savez(
        "results/metrics/eval_baseline.npz",
        y_true=test_y,
        y_pred=test_pred,
        known_probs=test_probs,
        holdout_conf=holdout_conf,
        holdout_entropy=holdout_entropy,
        holdout_pred_dist=pred_dist,
    )
    summary = {
        "split": "70/15/15 time-ordered by flow",
        "overall_accuracy": round(test_acc, 4),
        "macro_f1": round(test_f1, 4),
        "ECE": round(test_ece, 4),
        "known_mean_max_softmax": round(float(test_probs.max(axis=1).mean()), 4),
        "heldout_mean_max_softmax": round(float(holdout_conf.mean()), 4),
        "heldout_mean_entropy": round(float(holdout_entropy.mean()), 4),
        "heldout_pred_dist": {cls: int(pred_dist[i]) for i, cls in enumerate(train_ds.classes)},
    }
    with open("results/metrics/eval_baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved held-out test metrics to results/metrics/eval_baseline_summary.json and eval_baseline.npz")


if __name__ == "__main__":
    train_model()