import sys, json, numpy as np, warnings, time
warnings.filterwarnings('ignore')
from semisynth import *
from train import prepare, fit_kF
pids = sys.argv[1].split(',')
out = {}
for i, pid in enumerate(pids):
    # reference statistics and k_F from the other patients only (patient-held-out)
    others = prepare([p for p in pids if p != pid])
    ref = SourceRef().fit(others)
    kF = {m: fit_kF(others, ref, dict(c, gate=('F',))) for m, c in MODES.items() if c.get('gate')}
    for seed in (0, 1):
        r = run_patient(pid, deltas=(0.05, 0.1, 0.2), seed=seed, ref=ref, kF=kF)
        if r is None: continue
        for (m, d), v in r.items(): out[f'{pid}|{seed}|{m}|{d}'] = v
    print(pid, 'done', time.strftime('%H:%M'), flush=True)
    json.dump(out, open('./results/semisynth2.json', 'w'))
