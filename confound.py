"""Leakage check: AUROC of 'time since segment start' as the only predictor of the preictal label."""
import json, numpy as np
from sklearn.metrics import roc_auc_score
from prep import load, labels
ds = {r[0]: r for r in json.load(open('./results/datasets.json'))}
out = {}
for p in sorted(ds):
    if not p.startswith('chb'): continue
    P = labels(load(p)); t, seg, y = P['t'], P['seg'], P['y']
    first = {s: t[seg == s].min() for s in np.unique(seg)}
    ts = t - np.array([first[s] for s in seg]); k = y >= 0
    if len(np.unique(y[k])) < 2: continue
    out[p] = dict(auc=float(roc_auc_score(y[k], ts[k])), complete=bool(ds[p][5] >= 24))
json.dump(out, open('./results/confound.json', 'w'), indent=1)
ex = [v['auc'] for v in out.values() if not v['complete']]; lo = [v['auc'] for v in out.values() if v['complete']]
print(len(ex), np.median(ex), sum(a >= 0.9 for a in ex), np.median(lo))
