"""Controls for the attack, on the SAME flows and model as the official run (batched, fast on a GPU).

  python scripts/attack_controls.py                                   # the model used for the report
  python scripts/attack_controls.py --ckpt results/metrics/resnet18_seed0.pth --tag seed0
  python scripts/attack_controls.py --cap 20 --steps 5                # quick smoke test

Arms (all evaluated in image space, same budgets as the official run, 8-bit quantised):
  payload         gradient attack, channel 0, payload pixels only          = the official constrained attack
  ch0_bytes       gradient attack, channel 0, ALL byte pixels (headers too)  = ignores the protocol constraint
  all_pixels      gradient attack, all three channels, every pixel           = fully unconstrained
  random_payload  random +-eps noise on the payload pixels (no gradient)     = does the optimisation matter?
  random_all      random +-eps noise on every pixel (no gradient)

Every selected flow is kept, including the flows with no editable payload (the official run skipped them), and
flagged with has_payload. The comparison that matters is on has_payload flows; the 'all' subset shows what an
unconstrained attacker can do to the flows the constraint makes unattackable.

Writes results/controls/controls_summary_<tag>.csv and controls_per_sample_<tag>.npz
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F

from audit_utils import (ATTACK_CLASSES, ALPHA_BYTES, EPS_BYTES, MODEL_PATH, P, B, SEED, STEPS,
                         flow_to_image, load_cache, official_flow_list, prepare, select_flows)

ARMS = [("payload", "payload", "pgd"), ("ch0_bytes", "ch0_bytes", "pgd"), ("all_pixels", "all", "pgd"),
        ("random_payload", "payload", "random"), ("random_all", "all", "random")]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / den, (c + h) / den


def load_model(path, device):
    try:
        ck = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(path, map_location=device)
    n = ck["state_dict"]["fc.weight"].shape[0]
    model = timm.create_model(ck.get("model", "resnet18"), pretrained=False, in_chans=ck.get("in_chans", 3), num_classes=n)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ck["class_to_idx"]


@torch.no_grad()
def predict(model, x, bs=512):
    probs = torch.cat([F.softmax(model(x[i:i + bs]), 1) for i in range(0, len(x), bs)])
    conf, pred = probs.max(1)
    return pred, conf


def build_inputs(sel):
    X, masks, meta = [], {"payload": [], "ch0_bytes": [], "all": []}, []
    for f in sel:
        pr = prepare(f)
        if pr is None:
            continue
        X.append(torch.from_numpy(flow_to_image(pr["pkts"]).astype(np.float32) / 255.0))
        m = torch.zeros(3, P, B); m[0] = torch.from_numpy(pr["payload"].astype(np.float32)); masks["payload"].append(m)
        m = torch.zeros(3, P, B); m[0] = torch.from_numpy(pr["any_byte"].astype(np.float32)); masks["ch0_bytes"].append(m)
        masks["all"].append(torch.ones(3, P, B))
        meta.append(dict(label=f["label"], has_payload=bool(pr["payload"].any()), editable=int(pr["payload"].sum())))
    return torch.stack(X), {k: torch.stack(v) for k, v in masks.items()}, meta


def pgd_batch(model, x0, y, mask, eps_pix, alpha_pix, steps):
    dp = torch.zeros_like(x0)
    for _ in range(steps):
        xa = (x0 + dp).clamp(0, 1).detach().requires_grad_(True)
        loss = F.cross_entropy(model(xa), y, reduction="sum")
        g, = torch.autograd.grad(loss, xa)
        dp = (dp + alpha_pix * g.sign() * mask).clamp(-eps_pix, eps_pix) * mask
    return ((x0 + dp).clamp(0, 1) * 255.0).round() / 255.0


def random_batch(x0, mask, eps_pix, gen):
    sign = torch.randint(0, 2, x0.shape, generator=gen, device=x0.device).float() * 2 - 1
    return ((x0 + sign * eps_pix * mask).clamp(0, 1) * 255.0).round() / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=MODEL_PATH)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--eps", type=int, nargs="*", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cap", type=int, default=None, help="flows per class (default: same as the official run)")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--official", default="results/metrics/adversarial_per_sample_OFFICIAL.npz")
    ap.add_argument("--out", default="results/controls")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eps_list, steps = args.eps or EPS_BYTES, args.steps or STEPS
    print(f"device {device} | model {args.ckpt} | budgets {eps_list} | steps {steps}")

    model, c2i = load_model(args.ckpt, device)
    benign = c2i["Benign"]
    _, flows = load_cache(args.cache)
    sel = select_flows(flows, cap=args.cap)
    X, masks, meta = build_inputs(sel)
    labels = np.array([m["label"] for m in meta])
    has_payload = np.array([m["has_payload"] for m in meta])
    editable = np.array([m["editable"] for m in meta])
    known = np.array([l in c2i for l in labels])
    X = X.to(device); masks = {k: v.to(device) for k, v in masks.items()}
    pred0, conf0 = predict(model, X)
    y = torch.tensor([c2i[l] if l in c2i else int(pred0[i]) for i, l in enumerate(labels)], device=device)   # unseen class: pseudo-label
    p0, y_np = pred0.cpu().numpy(), y.cpu().numpy()
    print(f"{len(X)} flows | with editable payload: {int(has_payload.sum())} | known-class: {int(known.sum())}")

    gen = torch.Generator(device=device); gen.manual_seed(SEED)
    res = {}
    for arm, mkind, method in ARMS:
        for e in eps_list:
            eps_pix, alpha = e / 255.0, min(ALPHA_BYTES, e) / 255.0
            preds, confs = [], []
            for i in range(0, len(X), args.bs):
                xb, yb, mb = X[i:i + args.bs], y[i:i + args.bs], masks[mkind][i:i + args.bs]
                xa = pgd_batch(model, xb, yb, mb, eps_pix, alpha, steps) if method == "pgd" else random_batch(xb, mb, eps_pix, gen)
                pr, cf = predict(model, xa)
                preds.append(pr.cpu()); confs.append(cf.cpu())
            res[(arm, e)] = (torch.cat(preds).numpy(), torch.cat(confs).numpy())
        print(f"  arm {arm} done")

    rows = []
    for (arm, e), (pred, conf) in res.items():
        for subset in ("has_payload", "all"):
            sm = has_payload if subset == "has_payload" else np.ones(len(labels), bool)
            for c in ATTACK_CLASSES:
                m = (labels == c) & sm
                if not m.any():
                    continue
                if c not in c2i:
                    k = int((pred[m] != p0[m]).sum()); n = int(m.sum()); lo, hi = wilson(k, n)
                    rows.append([args.tag, arm, e, subset, c, "pred_changed", n, "", k, "", k / n, lo, hi])
                else:
                    ok = m & (p0 == y_np)
                    fl = ok & (pred != p0)
                    n_ok, k = int(ok.sum()), int(fl.sum())
                    lo, hi = wilson(k, n_ok)
                    rows.append([args.tag, arm, e, subset, c, "flip", int(m.sum()), n_ok, k, int((fl & (pred == benign)).sum()),
                                 (k / n_ok) if n_ok else float("nan"), lo, hi])
    with open(out / f"controls_summary_{args.tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tag", "arm", "eps", "subset", "class", "kind", "n", "n_correct_before", "flips", "to_benign", "rate", "ci_lo", "ci_hi"])
        w.writerows(rows)
    arrs = dict(label=labels, known=known, has_payload=has_payload, editable_pixels=editable,
                clean_pred=p0, clean_conf=conf0.cpu().numpy())
    for (arm, e), (pred, conf) in res.items():
        arrs[f"{arm}_{e}_pred"], arrs[f"{arm}_{e}_conf"] = pred, conf
    np.savez(out / f"controls_per_sample_{args.tag}.npz", **arrs)

    for subset, title in (("has_payload", "flows WITH editable payload (the comparable set)"), ("all", "ALL sampled flows (incl. those with no editable payload)")):
        print(f"\nFlip rates, {title}   [flips / flows correct before the attack]")
        for c in ("DDoS-UDP_Flood", "Recon-PortScan", "DoS-TCP_Flood"):
            print(f"  {c}")
            for arm, _, _ in ARMS:
                cells = []
                for e in eps_list:
                    r = [x for x in rows if x[1] == arm and x[2] == e and x[3] == subset and x[4] == c]
                    cells.append(f"eps{e}: {r[0][8]}/{r[0][7]}" if r else f"eps{e}: -")
                print(f"    {arm:<15}" + "  ".join(cells))

    if Path(args.official).exists():
        o = np.load(args.official, allow_pickle=True)
        oe = sorted({int(k.split("_")[0]) for k in o.files if k.split("_")[0].isdigit()})
        olab = o[f"{oe[0]}_label"].astype(str)
        if len(olab) == int(has_payload.sum()) and np.array_equal(olab, labels[has_payload]):
            print("\nSanity check, 'payload' arm (image space) vs the official run (after packet realisation):")
            op0 = o[f"{oe[0]}_clean_pred"].astype(int)
            for e in [x for x in oe if x in eps_list]:
                oa = o[f"{e}_adv_pred"].astype(int)
                for c in ("DDoS-UDP_Flood", "Recon-PortScan"):
                    mo = olab == c
                    ok_o = mo & (op0 == c2i[c])
                    mine = (labels == c) & has_payload
                    ok_m = mine & (p0 == y_np)
                    print(f"  eps {e:>2} {c:<16} official {int((ok_o & (oa != op0)).sum())}/{int(ok_o.sum())}   here {int((ok_m & (res[('payload', e)][0] != p0)).sum())}/{int(ok_m.sum())}")
            print("  (small differences are expected: the official run used a different batching order; large ones mean a setting differs)")
        else:
            print("\n[warning] could not align with the official per-sample file (different number or order of flows); skipped the sanity check.")
    print(f"\nwrote {out}/controls_summary_{args.tag}.csv and controls_per_sample_{args.tag}.npz")


if __name__ == "__main__":
    main()
