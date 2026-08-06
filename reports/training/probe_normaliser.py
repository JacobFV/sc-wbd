"""Is the heavy tail a corpus property, or a defect in the normaliser?

SimCorpus divides by scale = median_over_regions(std_over_time), floored at 1e-6.
If most regions are near-silent in a window, that median is tiny and z explodes.
"""
import json
from collections import defaultdict
import h5py, numpy as np

d = json.load(open("/data/scwbd/sim_corpus/index_fast.json"))
W, PER = 72, 24
rec = []
for s in d["shards"]:
    with h5py.File(s["path"], "r") as f:
        a = f["activity"]
        for i in np.linspace(0, a.shape[0]-1, PER).astype(int):
            x = np.asarray(a[i, :W], dtype=np.float32)
            sd = x.std(0) + 1e-6
            scale = float(np.median(sd))
            z = (x - x.mean(0, keepdims=True)) / max(scale, 1e-6)
            rec.append((s["backend"], scale, float(np.median(sd)), float(sd.max()),
                        float(z.std()), float((sd < 1e-5).mean())))

import numpy as np
zs = np.array([r[4] for r in rec]); med = np.median(zs)
extreme = [r for r in rec if r[4] > 10*med]
normal  = [r for r in rec if r[4] <= 10*med]
print(f"windows: {len(rec)}   extreme (z-std >10x median): {len(extreme)} = {100*len(extreme)/len(rec):.1f}%")
print()
print(f"{'group':10s} {'median regional sd':>20s} {'max regional sd':>17s} {'frac regions ~silent':>21s}")
for name, grp in (("normal", normal), ("EXTREME", extreme)):
    if grp:
        print(f"{name:10s} {np.median([g[2] for g in grp]):>20.3e} "
              f"{np.median([g[3] for g in grp]):>17.3e} {np.mean([g[5] for g in grp]):>20.1%}")
print()
cnt = defaultdict(int); tot = defaultdict(int)
for r in rec:
    tot[r[0]] += 1
    if r[4] > 10*med: cnt[r[0]] += 1
print("extreme-window rate by backend:")
for b in sorted(tot, key=lambda k: -cnt[k]/max(tot[k],1)):
    print(f"  {b:16s} {cnt[b]:>3d}/{tot[b]:<3d} = {100*cnt[b]/tot[b]:5.1f}%")
print()
hit = sum(1 for r in rec if r[1] <= 1e-6*1.001)
print(f"windows where the 1e-6 floor actually bound (median sd <= 1e-6): {hit} = {100*hit/len(rec):.1f}%")
