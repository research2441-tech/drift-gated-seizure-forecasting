"""Controlled preictal-preservation experiment on real interictal EEG features.

A slow, focal spectral-shape change (theta up, alpha down, ramping over the occurrence period)
is written into real interictal stretches before pseudo-onsets; recording disturbances
(channel gain step, EMG burst, channel loss) are written elsewhere. Each adaptation procedure is
run causally and model-free; the reference-relative input is then scored with a matched filter
for the injected pattern. Retention = how much of the injected contrast survives adaptation;
nuisance response = how often a disturbance produces a contrast as large as a true change."""
import numpy as np, json, os
from sklearn.metrics import roc_auc_score
from config import *
from prep import load, labels
from source_ref import SourceRef
from deploy import run

THETA, ALPHA_B = [2, 3], [4, 5]          # 4-8 Hz and 8-13 Hz band indices
FOCAL = [0, 1, 2, 3, 4, 5]               # left temporal/parasagittal chain

def pattern():
    """theta increase / alpha decrease, balanced so that it carries no broadband (gain) component"""
    v = np.zeros(16); v[THETA] = 1.0; v[ALPHA_B] = -1.0
    return v

def inject(P, delta, rng, n_events=6):
    """returns a modified copy with pseudo-onsets and disturbance intervals (all on y==0 time)"""
    Q = dict(P); x = P['x'].copy(); t = P['t']; seg = P['seg']
    ok = (P['y'] == 0)
    onsets, dist = [], []
    b = np.r_[0, np.flatnonzero(np.diff(seg)) + 1, len(seg)]
    cand = []
    for s, e in zip(b[:-1], b[1:]):
        if (t[e - 1] - t[s]) < 4 * 3600: continue
        for o in np.arange(t[s] + 2.5 * 3600, t[e - 1] - 600, 4 * 3600):
            i = np.searchsorted(t, o)
            lo = np.searchsorted(t, o - 3 * 3600)
            if ok[lo:i + 1].all() and seg[lo] == seg[i]: cand.append(o)
    rng.shuffle(cand); onsets = sorted(cand[:n_events])
    pat = pattern()
    for o in onsets:
        sel = (t > o - SPH - SOP) & (t <= o)
        ramp = np.clip((t[sel] - (o - SPH - SOP)) / SOP, 0, 1)
        x[np.ix_(np.flatnonzero(sel), FOCAL)] += (delta * ramp)[:, None, None] * pat
    # disturbances at the midpoint between pseudo-onsets (or far before first)
    taken = [(o - 3 * 3600, o + 600) for o in onsets]
    def free(a, z): return all(z < u or a > v for u, v in taken) and ok[(t >= a) & (t <= z)].all() and len(np.unique(seg[(t >= a) & (t <= z)])) == 1
    kinds = []
    for o in np.arange(t[0] + 1800, t[-1] - 7200, 2400):
        if len(kinds) >= 3 * max(1, len(onsets)) // 2: break
        if not free(o, o + 3600): continue
        k = ['gain', 'emg', 'loss'][len(kinds) % 3]
        sel = (t > o) & (t <= o + 3600); idx = np.flatnonzero(sel); ch = FOCAL
        if k == 'gain': x[np.ix_(idx, ch)] += 0.30 * np.r_[[2.0] * 13, 1.0, 1.0, 0.0]
        elif k == 'emg':
            emg = np.array([-2, -2, -2, -2, -1.5, -1, -0.5, 0, 0.3, 0.5, 0.7, 0.8, 1.0]) + 1.2
            x[np.ix_(idx, ch, np.arange(13))] = np.log10(10 ** x[np.ix_(idx, ch, np.arange(13))] + 10 ** emg)
            x[np.ix_(idx, ch, [13])] += 0.25
        else:
            x[np.ix_(idx, ch[:3], np.arange(13))] = -9.0; x[np.ix_(idx, ch[:3], [13])] = -6.0
            x[np.ix_(idx, ch[:3], [14])] = -3.0; x[np.ix_(idx, ch[:3], [15])] = 1.0
        taken.append((o - 600, o + 4200)); kinds.append((k, o))
    Q['x'] = x
    return Q, onsets, kinds

def score(inp, usable):
    """matched filter on the reference-relative input of the focal channels"""
    rel = inp[:, FOCAL, 16:32].astype(np.float64)
    u = usable[:, FOCAL]
    pat = pattern() / np.linalg.norm(pattern())
    s = (rel * pat).sum(-1)
    return np.where(u.any(1), (s * u).sum(1) / np.maximum(u.sum(1), 1), np.nan)

MODES = {
    'STATIC':    dict(mode='zscore', gate=None, tau_a=1e12, freeze_after=900),
    'ALIGN':     dict(mode='zscore', gate=None),
    'SLOWALIGN': dict(mode='zscore', gate=None, tau_a=21600.0, tau_ref=21600.0),
    'ALIGN+G':   dict(mode='zscore', gate=('F',)),
    'GAIN':      dict(mode='gain', gate=None, recentre=True, tau_ref=21600.0),
    'DRIFT':     dict(mode='gain', gate=('F',), recentre=True, tau_ref=21600.0),
}

def run_patient(pid, deltas=(0.05, 0.1, 0.2), seed=0, ref=None, kF=None):
    P = labels(load(pid)); rng = np.random.default_rng(seed)
    out = {}
    for dlt in deltas:
        Q, onsets, kinds = inject(P, dlt, np.random.default_rng(seed))
        if not onsets: return None
        t = Q['t']
        late = np.zeros(len(t), bool); ctrl = np.zeros(len(t), bool)
        for o in onsets:
            late |= (t > o - SPH - 600) & (t <= o - SPH)               # last 10 min of the occurrence period
            ctrl |= (t > o - 3 * 3600) & (t <= o - 2 * 3600)           # same stream, 2-3 h earlier
        dmask = {k: np.zeros(len(t), bool) for k in ('gain', 'emg', 'loss')}
        for k, o in kinds: dmask[k] |= (t > o) & (t <= o + 3600)
        for name, cfg in MODES.items():
            c = dict(cfg)
            if c.get('gate'): c['k_F'] = kF[name] if isinstance(kF, dict) else kF
            o_ = run(Q, None, ref, c, keep_inputs=True)
            s = score(o_['inp'], o_['usable'])
            av = ~np.isnan(s) & (o_['state'] > 0)
            sc = s[ctrl & av]; sl = s[late & av]
            res = dict(retained=float(np.nanmean(sl) - np.nanmean(sc)),
                       auc=float(roc_auc_score(np.r_[np.zeros(len(sc)), np.ones(len(sl))], np.r_[sc, sl])) if len(sc) and len(sl) else np.nan,
                       hold_late=float((o_['state'][late] == 1).mean()))
            thr = np.nanquantile(sc, 0.95) if len(sc) else np.nan
            for k, m in dmask.items():
                if m.any():
                    res['fp_' + k] = float(np.nanmean((s[m & av] > thr))) if (m & av).any() else 0.0
                    res['unav_' + k] = float((o_['state'][m] == 0).mean())
                    res['hold_' + k] = float((o_['state'][m] == 1).mean())
            out[(name, dlt)] = res
    return out
