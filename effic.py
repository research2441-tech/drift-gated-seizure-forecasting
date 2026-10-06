import json, glob, os, numpy as np, torch, time, resource
from model import Expert
from config import RES
out = {}
e = Expert(); out['params_per_expert'] = int(sum(p.numel() for p in e.parameters()))
tt = collections = {}
res = {}
for vdir in glob.glob(RES + 'runs/chb_f*_s*/*/'):
    v = os.path.basename(vdir.rstrip('/'))
    m = json.load(open(vdir + 'meta.json')) if os.path.exists(vdir + 'meta.json') else None
    if not m: continue
    for f in glob.glob(vdir + '*.npz'):
        z = np.load(f); n = len(z['state'])
        res.setdefault(v, []).append((float(z['sec']), n))
    res.setdefault(v + '_train', []).append(m['t_train'])
for v, L in res.items():
    if v.endswith('_train'): out[v] = float(np.mean(L))
    else:
        sec = sum(a for a, _ in L); n = sum(b for _, b in L)
        out[v + '_ms_per_step'] = 1000 * sec / (n / 15); out[v + '_x_realtime'] = (n * 4.0) / sec
json.dump(out, open(RES + 'effic.json', 'w'), indent=1); print(json.dumps(out, indent=1)[:1500])
