# Save as scripts/transfer.py
import csv
import numpy as np
import torch
from pathlib import Path
from attack_controls import build_inputs, load_model, pgd_batch, predict
from audit_utils import ALPHA_BYTES, ATTACK_CLASSES, STEPS, load_cache, select_flows

MODELS = [
    ("base", "results/metrics/resnet18_pcap_model.pth"),
    ("seed0", "results/metrics/resnet18_seed0.pth"),
    ("seed1", "results/metrics/resnet18_seed1.pth"),
    ("seed2", "results/metrics/resnet18_seed2.pth"),
]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, flows = load_cache(None)
    sel = select_flows(flows, cap=150)
    X, masks, meta = build_inputs(sel)
    X = X.to(device)
    mask = masks["payload"].to(device)

    labels = np.array([m["label"] for m in meta])
    has_payload = np.array([m["has_payload"] for m in meta])
    known_mask = np.isin(labels, ATTACK_CLASSES) & has_payload

    loaded = {}
    clean_preds = {}
    for name, ckpt in MODELS:
        m, c2i = load_model(ckpt, device)
        loaded[name] = (m, c2i)
        y = torch.tensor([c2i.get(l, 0) for l in labels], device=device)
        pred, _ = predict(m, X)
        clean_preds[name] = (pred.cpu().numpy(), y)

    eps_pix = 32.0 / 255.0
    alpha_pix = ALPHA_BYTES / 255.0
    adv_examples = {}
    for src_name, _ in MODELS:
        m, _ = loaded[src_name]
        _, y = clean_preds[src_name]
        torch.manual_seed(42)
        adv_examples[src_name] = pgd_batch(m, X, y, mask, eps_pix, alpha_pix, STEPS)

    out_rows = []
    header = ["source \\ target"] + [dst for dst, _ in MODELS]
    print("\n4x4 Transfer Matrix (Flip Rate at eps=32 on editable known-class flows):")
    print("  ".join(f"{h:<14}" for h in header))

    for src_name, _ in MODELS:
        row = [src_name]
        x_adv = adv_examples[src_name]
        for dst_name, _ in MODELS:
            dst_m, _ = loaded[dst_name]
            p0, y_t = clean_preds[dst_name]
            y_np = y_t.cpu().numpy()
            adv_pred, _ = predict(dst_m, x_adv)
            adv_np = adv_pred.cpu().numpy()

            correct0 = known_mask & (p0 == y_np)
            flips = correct0 & (adv_np != p0)
            rate = flips.sum() / correct0.sum() if correct0.sum() > 0 else 0.0
            row.append(f"{flips.sum()}/{correct0.sum()} ({rate:.3f})")
            out_rows.append({"source": src_name, "target": dst_name, "flips": int(flips.sum()), "n_correct": int(correct0.sum()), "flip_rate": round(float(rate), 4)})
        print("  ".join(f"{c:<14}" for c in row))

    out_csv = Path("results/controls/transfer_matrix_eps32.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "target", "flips", "n_correct", "flip_rate"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nSaved {out_csv}")

if __name__ == "__main__":
    main()