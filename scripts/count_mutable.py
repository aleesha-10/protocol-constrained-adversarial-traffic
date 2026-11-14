import pickle, collections
raw = pickle.load(open("data/flows_cache.pkl", "rb"))

def unpack(f, label=None):
    if isinstance(f, dict):
        return f.get("label", label), f["pkts"], f["pstarts"]
    lab, pk, ps = label, None, None
    for item in f:
        if isinstance(item, str):
            lab = item
        elif isinstance(item, (list, tuple)) and item:
            first = item[0]
            if isinstance(first, (tuple, list)) and len(first) == 2 and isinstance(first[1], (bytes, bytearray)):
                pk = item
            elif isinstance(first, int):
                ps = item
    return lab, pk, ps

if isinstance(raw, dict):
    items = [(lab, f) for lab, lst in raw.items() for f in lst]
else:
    items = [(None, f) for f in raw]

lab, pk, ps = unpack(items[0][1], items[0][0])
print("detected -> label:", lab, "| packets:", None if pk is None else len(pk),
      "| pstarts:", None if ps is None else len(ps))
assert pk is not None and ps is not None and len(pk) == len(ps), "detection failed"

c = collections.Counter()
for lab0, f in items:
    lab, pk, ps = unpack(f, lab0)
    mut = sum(max(0, min(64, len(b)) - p) for (_, b), p in zip(pk, ps))
    c[(lab, "zero-mutable" if mut == 0 else "editable")] += 1
for k in sorted(c, key=str):
    print(k, c[k])