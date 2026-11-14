"""Evaluate the trained PCAP-image classifier.

Part A (closed set): metrics on the classes seen during training.
Part B (open set):   behaviour on the held-out class(es) the model never saw.

All class metadata comes from the checkpoint written by train.py.
No hardcoded class lists, no hardcoded head sizes.
"""
import json
import sys
from pathlib import Path


root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import timm
from sklearn.metrics import classification_report, confusion_matrix

from src.data.dataset import PCAPImageDataset
from torch.utils.data import DataLoader, Dataset, Subset

class RemappedSubset(Dataset):
    """View of `base` over `indices`, with labels remapped through label_map."""
    def __init__(self, base, indices, label_map):
        self.base, self.indices, self.label_map = base, indices, label_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        return x, self.label_map[int(y)]


def evaluate_model():
    data_dir = "data/raw"
    model_path = "results/metrics/resnet18_pcap_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Checkpoint: metadata + weights (new format from train.py)
    ckpt = torch.load(model_path, map_location=device)
    assert "state_dict" in ckpt and "class_to_idx" in ckpt, (
        "Old-format checkpoint (raw state_dict). Re-run train.py so the "
        "checkpoint contains metadata, or update this script."
    )
    state_dict = ckpt["state_dict"]
    saved_class_to_idx = ckpt["class_to_idx"]
    arch_name = ckpt.get("model", "resnet18")
    in_chans = ckpt.get("in_chans", 3)
    num_classes = state_dict["fc.weight"].shape[0]

    # Class names in training-index order (derived, not hardcoded)
    known_names = [n for n, _ in sorted(saved_class_to_idx.items(), key=lambda kv: kv[1])]
    assert num_classes == len(known_names), "checkpoint head size != class_to_idx size"
    print(f"Checkpoint: {arch_name}, in_chans={in_chans}, {num_classes} classes")
    print(f"Known classes: {known_names}")

    # 2. Full dataset ONCE (all classes, including hold-outs)
    print("Loading full dataset (including hold-outs)...")
    full_view = PCAPImageDataset(root_dir=data_dir)
    full_c2i = full_view.class_to_idx

    missing = [n for n in known_names if n not in full_c2i]
    assert not missing, f"Checkpoint classes missing from dataset: {missing}"
    holdout_names = sorted(set(full_c2i) - set(known_names))
    print(f"Hold-out classes: {holdout_names if holdout_names else 'none'}")

    # Match by NAME. Full-view label indices differ from training indices,
    # so remap full-view label -> training label for the known subset.
    remap = {full_c2i[n]: train_idx for n, train_idx in saved_class_to_idx.items()}
    holdout_labels = {full_c2i[n] for n in holdout_names}

    if getattr(full_view, "targets", None) is not None:
        targets = full_view.targets
    elif hasattr(full_view, "samples"):
        targets = [int(s[1]) for s in full_view.samples]
    else:
        targets = [int(y) for _, y in full_view]

    known_idx, holdout_idx = [], []
    for i, y in enumerate(targets):
        y = int(y)
        if y in remap:
            known_idx.append(i)
        elif y in holdout_labels:
            holdout_idx.append(i)
    print(f"Closed-set samples: {len(known_idx)} | Hold-out samples: {len(holdout_idx)}")

    known_loader = DataLoader(RemappedSubset(full_view, known_idx, remap),
                      batch_size=32, shuffle=False)
    holdout_loader = (DataLoader(Subset(full_view, holdout_idx), batch_size=32, shuffle=False)
                      if holdout_idx else None)

    # 3. Model from checkpoint metadata
    model = timm.create_model(arch_name, pretrained=False, in_chans=in_chans,
                              num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # 4. Part A: closed-set evaluation on training classes
    y_true, y_pred, probs_all = [], [], []
    print("\nRunning closed-set evaluation...")
    with torch.no_grad():
        for x, y in known_loader:
            probs = F.softmax(model(x.to(device)), dim=1).cpu()
            probs_all.append(probs)
            y_pred.extend(probs.argmax(1).numpy())
            y_true.extend(np.asarray(y).astype(int).tolist())
    probs_all = torch.cat(probs_all)
    known_conf = probs_all.max(1).values.numpy()

    labels_range = list(range(num_classes))
    print("\n--- Closed-set report (training classes) ---")
    print(classification_report(y_true, y_pred, labels=labels_range,
                                target_names=known_names, digits=4))
    print("Confusion matrix (rows = true, cols = pred):")
    print(confusion_matrix(y_true, y_pred, labels=labels_range))
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    print(f"Overall accuracy: {acc:.4f}")

    # 5. Part B: open-set probe on unseen class(es)
    hold_conf, hold_entropy, pred_dist = [], [], np.zeros(num_classes)
    if holdout_loader:
        print("\nRunning open-set probe on unseen traffic...")
        with torch.no_grad():
            for x, _ in holdout_loader:
                probs = F.softmax(model(x.to(device)), dim=1).cpu()
                conf, pred = probs.max(1)
                hold_conf.extend(conf.numpy())
                hold_entropy.extend(-(probs * probs.clamp_min(1e-9).log()).sum(1).numpy())
                for p in pred.numpy():
                    pred_dist[p] += 1

        n = max(len(hold_conf), 1)
        print(f"\n--- Unseen-class probe ({len(hold_conf)} samples: {holdout_names}) ---")
        print(f"Mean max-softmax: {np.mean(hold_conf):.4f}  (known classes: {known_conf.mean():.4f})")
        print(f"Mean predictive entropy: {np.mean(hold_entropy):.4f}")
        print("Where unseen attack traffic gets absorbed:")
        for i, c in enumerate(known_names):
            print(f"  -> {c}: {int(pred_dist[i])} ({pred_dist[i] / n:.1%})")

    # 6. Persist for Phase 3 (plots, calibration analysis) without re-running
    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": str(model_path),
        "arch": arch_name,
        "class_to_idx": saved_class_to_idx,
        "holdout_classes": holdout_names,
        "closed_set_accuracy": acc,
        "closed_set_report": classification_report(
            y_true, y_pred, labels=labels_range, target_names=known_names,
            digits=4, output_dict=True),
        "holdout_mean_conf": float(np.mean(hold_conf)) if hold_conf else None,
        "holdout_mean_entropy": float(np.mean(hold_entropy)) if hold_entropy else None,
        "holdout_pred_distribution": {c: int(pred_dist[i]) for i, c in enumerate(known_names)},
    }
    (out_dir / "eval_baseline_summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(out_dir / "eval_baseline.npz",
             y_true=np.array(y_true), y_pred=np.array(y_pred),
             known_probs=probs_all.numpy(),
             holdout_conf=np.array(hold_conf), holdout_entropy=np.array(hold_entropy),
             holdout_pred_dist=pred_dist)
    print(f"\nSaved: {out_dir / 'eval_baseline_summary.json'} + eval_baseline.npz")


if __name__ == "__main__":
    evaluate_model()