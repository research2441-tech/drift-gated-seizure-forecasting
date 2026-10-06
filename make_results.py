"""Builds all result tables/figure data from stored runs. Writes results/summary.json."""
import json, numpy as np, collections, sys
from analyze import collect, pmean, paired, holm, boot_ci
from exp import get
from config import *

MAIN = ['SRC', 'HEAD', 'TENT', 'STATIC', 'ALIGN', 'ALIGN+G', 'GAIN', 'DRIFT']
ABL = ['DRIFT', 'DRIFT-noE', 'DRIFT-noF', 'DRIFT-noR', 'DRIFT-noQ', 'DRIFT-noHmax', 'DRIFT-noShrink', 'DRIFT-noHead']
METRICS = {'auc': ('auc', None), 's20': ('sens@tiw20', None), 's10': ('sens@tiw10', None),
           'sens': ('sens', 'ltt_online'), 'fa': ('fa_h', 'ltt_online'), 'tiw': ('tiw', 'ltt_online'),
           'lead': ('lead_min', 'ltt_online'), 'ece': ('ece', None), 'brier': ('brier', None), 'cov': ('coverage', None),
           'hold': ('hold', None)}

def per_patient(res, v, key, pol=None, pids=None):
    d = res[v]; P = pids or sorted(d)
    return np.array([pmean(d[p], key, pol) if p in d else np.nan for p in P])

def ioc(res, v, pids):
    s = per_patient(res, v, 'sens', 'ltt_online', pids); fa = per_patient(res, v, 'fa_h', 'ltt_online', pids)
    return s - (1 - np.exp(-fa * SOP / 3600))

def main(prefix='chb_f', out=RES + 'summary_chb.json'):
    res = collect(prefix)
    pids = sorted(set.intersection(*[set(res[v]) for v in MAIN if v in res]))
    S = {'pids': pids, 'n_events': {}, 'main': {}, 'tests': {}, 'abl': {}, 'calib': {}}
    for p in pids:
        S['n_events'][p] = res['DRIFT'][p][0]['policies']['ltt_online']['n_events']
    for v in MAIN:
        if v not in res: continue
        row = {}
        for m, (k, pol) in METRICS.items():
            x = per_patient(res, v, k, pol, pids); row[m] = [float(np.nanmean(x)), float(np.nanstd(x))]
            row[m + '_pp'] = [None if np.isnan(a) else float(a) for a in x]
        x = ioc(res, v, pids); row['ioc'] = [float(np.nanmean(x)), float(np.nanstd(x))]; row['ioc_pp'] = [None if np.isnan(a) else float(a) for a in x]
        # pooled event sensitivity at matched burden, weighted by events, bootstrap over patients
        S['main'][v] = row
    for m in ['auc', 's20', 'sens', 'fa', 'tiw', 'ece', 'ioc']:
        ps, rbs, ns, names = [], [], [], []
        for v in MAIN:
            if v == 'DRIFT' or v not in S['main']: continue
            p, rb, n = paired(S['main']['DRIFT'][m + '_pp'] and np.array(S['main']['DRIFT'][m + '_pp'], float), np.array(S['main'][v][m + '_pp'], float))
            ps.append(p if not np.isnan(p) else 1.0); rbs.append(rb); ns.append(n); names.append(v)
        adj = holm(ps)
        S['tests'][m] = {nm: dict(p=float(pp), p_holm=float(a), rb=rb, n=n) for nm, pp, a, rb, n in zip(names, ps, adj, rbs, ns)}
    for v in ABL:
        if v not in res: continue
        pp = sorted(res[v]); row = {}
        for m, (k, pol) in METRICS.items():
            a = per_patient(res, v, k, pol, pp); b = per_patient(res, 'DRIFT', k, pol, pp)
            row[m] = [float(np.nanmean(a)), float(np.nanmean(a - b))]
            if v != 'DRIFT':
                p, rb, n = paired(a, b); row[m + '_p'] = p
        S['abl'][v] = row
    # calibration validity: patients with >= 24 h interictal monitoring
    long = [p for p in pids if res['DRIFT'][p][0]['policies']['ltt_online']['inter_h'] >= 24]
    S['long'] = long
    for v in MAIN:
        if v not in res: continue
        S['calib'][v] = {}
        for pol in ['fixed05', 'winq', 'ltt_src', 'ltt_online']:
            t = per_patient(res, v, 'tiw', pol, long); f = per_patient(res, v, 'fa_h', pol, long); s = per_patient(res, v, 'sens', pol, long)
            S['calib'][v][pol] = dict(tiw_med=float(np.nanmedian(t)), tiw_pp=[float(a) for a in t], frac_ok=float(np.nanmean(t <= ALPHA + 1e-9)),
                                      fa=float(np.nanmean(f)), sens=float(np.nanmean(s)))
    json.dump(S, open(out, 'w'), indent=1)
    return S

if __name__ == '__main__':
    S = main(*(sys.argv[1:] if len(sys.argv) > 1 else []))
    for v, r in S['main'].items():
        print(v.ljust(8), ' '.join('%s=%.3f' % (m, r[m][0]) for m in ['auc', 's20', 'ioc', 'sens', 'fa', 'tiw', 'ece', 'cov']))
