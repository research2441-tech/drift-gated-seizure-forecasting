"""E3: disturbances injected into real target streams; trained experts from the seed-0 fold runs."""
import sys, os, json, glob, numpy as np, torch, warnings
warnings.filterwarnings('ignore')
from config import *
from exp import get
from model import Expert
from source_ref import SourceRef
from deploy import run
from alarm import policy, warning_cover
from disturb import apply, pick_times
torch.set_num_threads(1)
OUT = RES + 'disturb.json'
VAR = {'SRC': 'none', 'ALIGN': 'zscore', 'GAIN': 'gain', 'DRIFT': 'gain'}
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
runs = sorted(glob.glob(RES + 'runs/chb_f*_s1'))
for rd in runs:
    split = json.load(open(rd + '/split.json'))
    G = __import__('exp').subjects('chb')
    tr = [get(p) for g in split['tr'] for p in G[g]]
    ref = SourceRef().fit(tr)
    for v, tm in VAR.items():
        meta = json.load(open(rd + f'/{v}/meta.json')); cfg = meta['cfg']
        if cfg.get('gate'): cfg['gate'] = tuple(cfg['gate'])
        cfg.update(recal=False, head=False)
        ex = []
        for sd in torch.load(rd + f'/experts_{tm}.pt'):
            e = Expert(); e.load_state_dict(sd); e.eval(); ex.append(e)
        for g in split['tgt']:
            for pid in G[g]:
                P = get(pid)
                if (P['y'] == 0).sum() * WSEC < 12 * 3600: continue      # continuous recordings only
                for kind in ('gain', 'loss', 'emg'):
                    key = f'{pid}|{v}|{kind}'
                    if key in res: continue
                    ts = pick_times(P, n=1, rng=np.random.default_rng(hash(pid + kind) % 2**31))
                    if not ts: continue
                    t0 = ts[0]; lo = np.searchsorted(P['t'], t0 - 8 * 3600); hi = np.searchsorted(P['t'], t0 + 4 * 3600)
                    crop = {k: (val[lo:hi] if isinstance(val, np.ndarray) and len(val) == len(P['t']) else val) for k, val in P.items()}
                    out = {}
                    for tag, Q in (('clean', crop), ('dist', apply(crop, kind, t0))):
                        o = run(Q, ex, ref, cfg)
                        th = meta['theta_S']
                        al = policy(o['risk'], Q['t'], Q['seg'], th)
                        win = (Q['t'] > t0) & (Q['t'] <= t0 + (3 * 3600 if kind == 'gain' else 3600))
                        w = warning_cover(Q['t'], al)
                        out[tag] = dict(alarms=int(((al > t0) & (al <= Q['t'][win][-1])).sum()), tiw=float(w[win].mean()),
                                        unav=float((o['state'][win] == 0).mean()), hold=float((o['state'][win] == 1).mean()),
                                        risk=float(np.nanmean(o['risk'][win])))
                    res[key] = out
                    json.dump(res, open(OUT, 'w'))
                    print(key, out['clean']['tiw'], out['dist']['tiw'], flush=True)
