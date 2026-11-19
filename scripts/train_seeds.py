"""Train ResNet-18 with several seeds, using exactly the recipe of train.py (so the models are comparable).

  python scripts/train_seeds.py --seeds 0 1 2

Recipe copied from train.py: all windows of the known classes (hold-out DictionaryBruteForce excluded), ResNet-18
from scratch, Adam lr 1e-3, batch 32, 5 epochs, cross-entropy, deterministic cuDNN. Only the seed changes.
The window dataset is built once (slow, uses scapy) and cached in data/windows_cache.npz.
Saves results/metrics/resnet18_seed<N>.pth in the same checkpoint format as train.py, so evaluate.py and
attack_controls.py (--ckpt ... --tag seed<N>) work on them unchanged.
Like train.py this trains and reports on the same data: the seeds measure run-to-run variation of the TRAINED MODEL,
not generalisation.
"""
import argparse
import json
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
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

CACHE = Path("data/windows_cache.npz")
HOLDOUT = ["DictionaryBruteForce"]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_windows(data_dir):
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=False)
        return d["X"], d["y"], json.loads(str(d["class_to_idx"]))
    from src.data.dataset import PCAPImageDataset
    ds = PCAPImageDataset(root_dir=data_dir, exclude_classes=HOLDOUT)
    X = np.stack([s[0] for s in ds.samples]).astype(np.uint8)
    y = np.array([s[1] for s in ds.samples], dtype=np.int64)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, X=X, y=y, class_to_idx=np.array(json.dumps(ds.class_to_idx)))
    return X, y, ds.class_to_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, c2i = load_windows(args.data)
    classes = [n for n, _ in sorted(c2i.items(), key=lambda kv: kv[1])]
    print(f"{len(X)} windows | classes {c2i} | device {device}")
    Xt = torch.from_numpy(X.astype(np.float32) / 255.0)
    yt = torch.from_numpy(y)
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        set_seed(seed)
        gen = torch.Generator(); gen.manual_seed(seed)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=args.bs, shuffle=True, generator=gen)
        model = timm.create_model("resnet18", pretrained=False, in_chans=3, num_classes=len(classes)).to(device)
        opt, crit = optim.Adam(model.parameters(), lr=args.lr), nn.CrossEntropyLoss()
        model.train()
        for ep in range(args.epochs):
            tot = cor = 0; run = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(); out = model(xb); loss = crit(out, yb); loss.backward(); opt.step()
                run += loss.item() * len(xb); tot += len(xb); cor += (out.argmax(1) == yb).sum().item()
            print(f"  seed {seed} epoch {ep+1}/{args.epochs} loss {run/tot:.4f} train acc {100*cor/tot:.2f}%")
        path = f"results/metrics/resnet18_seed{seed}.pth"
        torch.save({"model": "resnet18", "in_chans": 3, "class_to_idx": c2i, "classes": classes, "state_dict": model.state_dict()}, path)
        print(f"  saved {path}")


if __name__ == "__main__":
    main()
