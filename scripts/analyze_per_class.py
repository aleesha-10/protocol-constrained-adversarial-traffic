import json, collections
import numpy as np
z   = np.load("results/metrics/adversarial_per_sample.npz", allow_pickle=True)
res = json.loads(open("results/metrics/adversarial_results.json", encoding="utf-8").read())
EPS = res.get("config", {}).get("eps_bytes", [1, 4, 16, 32])
C2I = {"Benign": 0, "DDoS-UDP_Flood": 1, "DoS-TCP_Flood": 2, "Recon-PortScan": 3}
for k, v in res.get("config", {}).items():
    if "class" in k.lower() and isinstance(v, dict) and v: C2I = {n: int(i) for n, i in v.items()}
E0  = str(EPS[0]); lab = z[f"{E0}_label"]
print(f"rows: {len(lab)}  counts: {dict(collections.Counter(lab))}\n")
hdr = f"{'class':24s}{'n':>5s}{'clean_acc':>10s}" + "".join(f" | e={e:<3d}evas  flip" for e in EPS)
print(hdr)
for cls, n in collections.Counter(lab).most_common():
    m, known, y = np.array([l == cls for l in lab]), cls in C2I, C2I.get(cls, -1)
    row = f"{cls:24s}{n:5d}"
    row += f"{(z[f'{E0}_clean_pred'][m] == y).mean():10.3f}" if known else f"{'n/a':>10s}"
    for e in EPS:
        ap, cp_ = z[f"{e}_adv_pred"][m], z[f"{e}_clean_pred"][m]
        row += f" | {(ap != y).mean():7.3f}" if known else " |     n/a"
        row += f" {(ap != cp_).mean():5.3f}"
    print(row)