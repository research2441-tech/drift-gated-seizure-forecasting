"""Event-level evaluation of stored risk streams under several threshold policies."""
import os, json, glob, numpy as np
from sklearn.metrics import roc_auc_score
from config import *
from exp import get
from alarm import *

def online_thetas(P, risk, theta_S, alpha=ALPHA, n_min=12, delta=0.10):
    """per-window threshold: theta_S until n_min target hour-blocks are adjudicated interictal,
    then the LTT threshold recomputed every hour from adjudicated target blocks only."""
    t = P['t']; inter = interictal_mask(t, P['onset'], P['offset'])
    warns = {th: warning_cover(t, policy(risk, t, P['seg'], th)) for th in THETAS}
    blocks = {th: dict(block_losses(t, inter, w)) for th, w in warns.items()}
    bids = sorted(blocks[THETAS[0]].keys())
    # a block b (hour b) is adjudicated once its end plus the warning horizon and the
    # post-ictal margin have elapsed
    th_t = np.full(len(t), theta_S if theta_S is not None else THETAS[0])
    hour = np.floor(t / 3600).astype(int)
    cur = th_t[0]; trace = []
    for h in np.unique(hour):
        now = h * 3600.0
        adj = [b for b in bids if (b + 1) * 3600 + WARN + POST_GAP <= now]
        if len(adj) >= n_min:
            L = [np.array([blocks[th][b] for b in adj]) for th in THETAS]
            c = ltt_threshold(L, alpha, delta)
            cur = c if c is not None else THETAS[0]
        th_t[hour == h] = cur; trace.append((int(h), float(cur), len(adj)))
    return th_t, trace

def ece(p, y, nb=10):
    b = np.minimum((p * nb).astype(int), nb - 1); e = 0.0
    for i in range(nb):
        s = b == i
        if s.any(): e += s.mean() * abs(p[s].mean() - y[s].mean())
    return e

def evaluate_stream(P, z, meta, alpha=ALPHA):
    risk = z['risk'].astype(np.float64); t = P['t']; res = {}
    lab = (P['y'] >= 0) & ~np.isnan(risk)
    y = P['y'][lab]; r = risk[lab]
    res['auc'] = float(roc_auc_score(y, r)) if 0 < y.mean() < 1 else np.nan
    a, b = meta['platt']; rr = np.clip(r, 1e-4, 1 - 1e-4); pc = 1 / (1 + np.exp(-(a * np.log(rr / (1 - rr)) + b)))
    res['ece'] = ece(pc, y); res['brier'] = float(np.mean((pc - y) ** 2))
    res['coverage'] = float((~np.isnan(risk)).mean())
    st = z['state']; res['hold'] = float((st == 1).mean()); res['unavail'] = float((st == 0).mean())
    pol = {}
    pol['fixed05'] = metrics(policy(risk, t, P['seg'], 0.5), t, P['onset'], P['offset'], P['lead'], risk)
    pol['winq'] = metrics(policy(risk, t, P['seg'], meta['theta_win']), t, P['onset'], P['offset'], P['lead'], risk)
    thS = meta['theta_S'] if meta['theta_S'] is not None else THETAS[0]
    pol['ltt_src'] = metrics(policy(risk, t, P['seg'], thS), t, P['onset'], P['offset'], P['lead'], risk)
    pol['ltt_online'] = metrics(policy_online(risk, t, P['seg'], z['theta'].astype(np.float64)), t, P['onset'], P['offset'], P['lead'], risk)
    res['n_updates'] = int(z['n_updates'])
    # matched-burden curve (oracle sweep, ranking quality only)
    curve = []
    for th in THETAS:
        m = metrics(policy(risk, t, P['seg'], th), t, P['onset'], P['offset'], P['lead'])
        curve.append((float(th), m['tiw'], m['sens'], m['fa_h']))
    res['curve'] = curve
    for target in (0.10, 0.20, 0.30):
        ok = [c for c in curve if c[1] <= target + 1e-9 and not np.isnan(c[2])]
        res[f'sens@tiw{int(target*100)}'] = max([c[2] for c in ok]) if ok else 0.0
    res['policies'] = pol
    return res

def evaluate_tag(tag, root=RES + 'streams/'):
    meta = json.load(open(root + tag + '/meta.json'))
    out = {}
    for f in sorted(glob.glob(root + tag + '/*.npz')):
        pid = os.path.basename(f)[:-4]
        out[pid] = evaluate_stream(get(pid), np.load(f), meta)
    json.dump(out, open(root + tag + '/eval.json', 'w'))
    return out
