"""Aggregate stored streams into per-patient metrics, tables and paired statistics."""
import os, json, glob, numpy as np, collections, warnings
from scipy.stats import wilcoxon
from config import *
from exp import get
from evaluate import evaluate_stream
warnings.filterwarnings('ignore')
ROOT = RES + 'runs/'

def collect(prefix, variants=None, cache=RES + 'eval_cache/'):
    """returns {variant: {pid: [metrics per seed]}}"""
    os.makedirs(cache, exist_ok=True)
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for run in sorted(glob.glob(ROOT + prefix + '*')):
        for vdir in sorted(glob.glob(run + '/*/')):
            v = os.path.basename(vdir.rstrip('/'))
            if variants and v not in variants: continue
            if not os.path.exists(vdir + 'meta.json'): continue
            meta = json.load(open(vdir + 'meta.json'))
            for f in sorted(glob.glob(vdir + '*.npz')):
                pid = os.path.basename(f)[:-4]
                ck = cache + os.path.basename(run) + '_' + v + '_' + pid + '.json'
                if os.path.exists(ck) and os.path.getmtime(ck) > os.path.getmtime(f):
                    r = json.load(open(ck))
                else:
                    r = evaluate_stream(get(pid), np.load(f), meta)
                    json.dump(r, open(ck, 'w'), default=float)
                r['seed'] = meta['seed']; out[v][pid].append(r)
    return out

def pmean(L, key, pol=None):
    vals = [(x['policies'][pol][key] if pol else x[key]) for x in L]
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else np.nan

def table(res, keys, pids=None):
    rows = {}
    for v, d in res.items():
        P = pids or sorted(d)
        rows[v] = {name: [pmean(d[p], k, pol) if p in d else np.nan for p in P] for name, (k, pol) in keys.items()}
    return rows

def holm(ps):
    ps = np.array(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * ps[j]); adj[j] = min(1, run)
    return adj

def paired(a, b):
    a, b = np.array(a), np.array(b); ok = ~np.isnan(a) & ~np.isnan(b); a, b = a[ok], b[ok]
    d = a - b; nz = d[d != 0]
    if len(nz) < 3: return np.nan, np.nan, len(a)
    p = wilcoxon(a, b, zero_method='wilcox').pvalue
    r = np.argsort(np.argsort(np.abs(nz))) + 1
    rb = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()
    return float(p), float(rb), len(a)

def boot_ci(vals, n=10000, seed=0):
    v = np.array([x for x in vals if not np.isnan(x)]); rng = np.random.default_rng(seed)
    bs = rng.choice(v, (n, len(v))).mean(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
