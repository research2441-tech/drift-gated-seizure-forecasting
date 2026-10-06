"""Patient-held-out experiment driver. One call = one (variant, fold, seed)."""
import os, json, time, numpy as np, torch
from config import *
from prep import load, labels
from source_ref import SourceRef
from train import source_inputs, fit_kF, train_ensemble
from deploy import run
from alarm import *
from variants import VARIANTS
torch.set_num_threads(2)

_cache = {}
def get(pid):
    if pid not in _cache: _cache[pid] = labels(load(pid))
    return _cache[pid]

def subjects(dataset):
    s = json.load(open(DATA + 'summary.json'))
    if dataset == 'chb':
        pids = sorted(p for p in s if p.startswith('chb'))
        groups = {}
        for p in pids: groups.setdefault('chb01' if p == 'chb21' else p, []).append(p)
    else:
        groups = {p: [p] for p in sorted(p for p in s if p.startswith('PN'))}
    return groups

def folds(dataset, k=4):
    g = subjects(dataset); s = json.load(open(DATA + 'summary.json'))
    order = sorted(g, key=lambda q: -sum(s[p]['n_seizures'] for p in g[q]))
    return [[order[i] for i in range(f, len(order), k)] for f in range(k)], g

def thresholds_from(streams, q_E=0.95, q_R=0.90, q_U=0.995):
    de = np.concatenate([o['DE'][(P['y'] == 0) & (o['state'] > 0)] for P, o in streams])
    rk = np.concatenate([o['risk'][(P['y'] == 0) & (o['state'] > 0)] for P, o in streams])
    de = de[~np.isnan(de)]; rk = rk[~np.isnan(rk)]
    return dict(k_E=float(np.quantile(de, q_E)), k_R=float(np.quantile(rk, q_R)), k_U=float(np.quantile(de, q_U)))

def platt(streams):
    """prevalence-preserving logistic recalibration of the ensemble risk on calibration patients"""
    from sklearn.linear_model import LogisticRegression
    z, y = [], []
    for P, o in streams:
        sel = (P['y'] >= 0) & ~np.isnan(o['risk'])
        r = np.clip(o['risk'][sel], 1e-4, 1 - 1e-4); z.append(np.log(r / (1 - r))); y.append(P['y'][sel])
    lr = LogisticRegression(C=1e3).fit(np.concatenate(z)[:, None], np.concatenate(y))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])

def source_theta(streams, alpha=ALPHA):
    losses = [[] for _ in THETAS]
    for P, o in streams:
        inter = interictal_mask(P['t'], P['onset'], P['offset'])
        for j, th in enumerate(THETAS):
            warn = warning_cover(P['t'], policy(o['risk'], P['t'], P['seg'], th))
            losses[j] += [l for _, l in block_losses(P['t'], inter, warn)]
    return ltt_threshold(losses, alpha), losses

def run_one(variant, dataset, fold, seed, k=4, out_dir=RES + 'streams/', steps=1200, K=3,
            train_groups=None, target_groups=None, tag=None):
    tag = tag or f'{variant}_{dataset}_f{fold}_s{seed}'
    os.makedirs(out_dir + tag, exist_ok=True)
    if os.path.exists(out_dir + tag + '/meta.json'): return
    t0 = time.time()
    if train_groups is None:
        F, G = folds(dataset, k)
        tgt_groups = F[fold]; src_groups = [g for f in F for g in f if g not in tgt_groups]
    else:
        G = {**subjects('chb'), **subjects('siena')}
        tgt_groups, src_groups = target_groups, train_groups
    rng = np.random.default_rng(100 * fold + seed)
    src_groups = list(rng.permutation(src_groups))
    n_cal = max(2, len(src_groups) // 5)
    cal_groups, tr_groups = src_groups[:n_cal], src_groups[n_cal:]
    tr = [get(p) for g in tr_groups for p in G[g]]; cal = [get(p) for g in cal_groups for p in G[g]]
    tg = [get(p) for g in tgt_groups for p in G[g]]
    ref = SourceRef().fit(tr)
    cfg = dict(VARIANTS[variant])
    base = dict(cfg); base['gate'] = ('F',) if cfg.get('gate') else None
    if cfg.get('gate'): base['k_F'] = cfg['k_F'] = fit_kF(tr, ref, base)
    data = source_inputs(tr, ref, base)
    experts = train_ensemble(data, K=K, seed=seed, steps=steps)
    del data
    # calibration patients: thresholds for the gate (ungated pass), then the full pass
    c0 = dict(cfg); c0.update(gate=None, head=False, recal=False)
    th = thresholds_from([(P, run(P, experts, ref, c0)) for P in cal])
    if cfg.get('gate'):
        cfg.update(th)                       # gated variants also own the explicit unavailable state
    else:
        cfg['q_min'] = 0.0                   # ungated baselines always emit a risk
    cal_streams = [(P, run(P, experts, ref, dict(cfg, head=False, recal=False))) for P in cal]
    theta_S, _ = source_theta(cal_streams)
    if theta_S is None: theta_S = float(THETAS[0])
    a, b = platt(cal_streams)
    rwin = np.concatenate([o['risk'][(P['y'] == 0) & ~np.isnan(o['risk'])] for P, o in cal_streams])
    theta_win = float(np.quantile(rwin, 1 - ALPHA))       # window-level quantile threshold
    cfg['theta_S'] = theta_S
    meta = dict(variant=variant, dataset=dataset, fold=fold, seed=seed, tr=tr_groups, cal=cal_groups,
                tgt=tgt_groups, cfg={k_: (v if not isinstance(v, tuple) else list(v)) for k_, v in cfg.items()},
                theta_S=theta_S, theta_win=theta_win, platt=[a, b])
    for P in tg:
        tt = time.time(); o = run(P, experts, ref, cfg); dt = time.time() - tt
        np.savez_compressed(out_dir + tag + f"/{P['pid']}.npz", risk=o['risk'].astype(np.float32),
                            src_risk=o['src_risk'].astype(np.float32), theta=o['theta'].astype(np.float32),
                            state=o['state'], DF=o['DF'].astype(np.float16), DE=o['DE'].astype(np.float16),
                            gamma=o['gamma'][::15].astype(np.float16), sec=dt, n_updates=o['n_updates'],
                            recal=np.array(o['recal_log']))
    torch.save([e.state_dict() for e in experts], out_dir + tag + '/experts.pt')
    meta['elapsed'] = time.time() - t0
    json.dump(meta, open(out_dir + tag + '/meta.json', 'w'), indent=1)
    print(tag, 'done %.0fs' % meta['elapsed'], 'theta_S', theta_S, flush=True)
