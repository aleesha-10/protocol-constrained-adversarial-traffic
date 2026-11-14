import numpy as np
d = np.load("results/metrics/adversarial_per_sample_OFFICIAL.npz", allow_pickle=True)
names = ["Benign", "DDoS-UDP_Flood", "DoS-TCP_Flood", "Recon-PortScan"]
eps = sorted({int(k.split("_")[0]) for k in d.files if k.split("_")[0].isdigit()})

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; den = 1 + z*z/n; c = p + z*z/(2*n)
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c-h)/den, (c+h)/den)

lab = d[f"{eps[0]}_label"].astype(str)
for cls in names[1:]:
    m = lab == cls
    cp = d[f"{eps[0]}_clean_pred"][m]
    ok = cp == names.index(cls)
    print(f"\n{cls}: n={m.sum()}, clean-correct={ok.sum()} ({ok.mean():.3f})")
    for e in eps:
        ap = d[f"{e}_adv_pred"][m]
        flip = (ok & (ap != cp)).sum()
        evade = (ok & (ap == 0)).sum()
        lo, hi = wilson(flip, ok.sum())
        print(f"  eps={e:>2}: flip {flip}/{ok.sum()} = {flip/max(ok.sum(),1):.3f} [{lo:.3f}, {hi:.3f}] | to-Benign {evade}")