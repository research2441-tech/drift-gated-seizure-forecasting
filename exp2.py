"""Patient-held-out experiment driver.
For one (fold, seed): experts are trained once per training procedure (input pipeline), and every
deployment variant that shares that pipeline reuses them, so gate and readout effects are isolated
at deployment time with identical weights."""
import os, json, time, sys, numpy as np, torch
from config import *
from exp import get, folds, subjects, thresholds_from, platt, source_theta
from source_ref import SourceRef
from train import source_inputs, fit_kF, train_ensemble
from deploy import run
from alarm import THETAS, ALPHA
torch.set_num_threads(int(os.environ.get('NT', '2')))

TRAIN = {'none':   dict(mode='none', gate=None, rel=False),
         'static': dict(mode='zscore', gate=None, tau_a=1e12, freeze_after=900),
         'zscore': dict(mode='zscore', gate=None),
         'zslow':  dict(mode='zscore', gate=None, tau_a=21600.0, tau_ref=21600.0),
         'gain':   dict(mode='gain', gate=None, recentre=True, tau_ref=21600.0)}
G = ('E', 'F', 'R')
DEPLOY = {
    'none':   {'SRC': dict(head=False), 'HEAD': dict(head=True), 'TENT': dict(head=True, tent=True)},
    'static': {'STATIC': dict(head=True)},
    'zscore': {'ALIGN': dict(head=True), 'ALIGN+G': dict(head=True, gate=G)},
    'zslow':  {'SLOWALIGN': dict(head=True)},
    'gain':   {'GAIN': dict(head=True), 'DRIFT': dict(head=True, gate=G),
               'DRIFT-noE': dict(head=True, gate=('F', 'R')), 'DRIFT-noF': dict(head=True, gate=('E', 'R')),
               'DRIFT-noR': dict(head=True, gate=('E', 'F')), 'DRIFT-noQ': dict(head=True, gate=G, use_quality=False),
               'DRIFT-noHmax': dict(head=True, gate=G, h_max=None), 'DRIFT-noShrink': dict(head=True, gate=G, kappa=1e-6),
               'DRIFT-noHead': dict(head=False, gate=G)},
}

def run_fold(dataset, fold, seed, k=4, steps=1000, K=3, out=RES + 'runs/', only=None, train_groups=None, target_groups=None, tag=None):
    tag = tag or f'{dataset}_f{fold}_s{seed}'
    os.makedirs(out + tag, exist_ok=True)
    if train_groups is None:
        F, Gr = folds(dataset, k)
        tgt_groups = F[fold]; src_groups = [g for f in F for g in f if g not in tgt_groups]
    else:
        Gr = {**subjects('chb'), **subjects('siena')}; tgt_groups, src_groups = target_groups, train_groups
    rng = np.random.default_rng(100 * fold + seed)
    src_groups = list(rng.permutation(src_groups)); n_cal = max(2, len(src_groups) // 5)
    cal_groups, tr_groups = src_groups[:n_cal], src_groups[n_cal:]
    tr = [get(p) for g in tr_groups for p in Gr[g]]; cal = [get(p) for g in cal_groups for p in Gr[g]]
    tg = [get(p) for g in tgt_groups for p in Gr[g]]
    ref = SourceRef().fit(tr)
    json.dump(dict(tr=tr_groups, cal=cal_groups, tgt=tgt_groups), open(out + tag + '/split.json', 'w'))
    for tm, tcfg in TRAIN.items():
        if only and tm not in only: continue
        todo = {v: d for v, d in DEPLOY[tm].items() if (seed == 0 or '-' not in v)}   # ablations: seed 0 only
        if all(os.path.exists(out + tag + f'/{v}/meta.json') for v in todo): continue
        t0 = time.time()
        kF = fit_kF(tr, ref, dict(tcfg, gate=('F',)))         # shape-change threshold for this pipeline
        data = source_inputs(tr, ref, tcfg)
        experts = train_ensemble(data, K=K, seed=seed, steps=steps); del data
        t_train = time.time() - t0
        c0 = dict(tcfg); c0.update(gate=None, head=False, recal=False)
        th = thresholds_from([(P, run(P, experts, ref, c0)) for P in cal])
        for v, dcfg in todo.items():
            if os.path.exists(out + tag + f'/{v}/meta.json'): continue
            os.makedirs(out + tag + '/' + v, exist_ok=True)
            cfg = dict(tcfg); cfg.update(dcfg)
            if cfg.get('gate'): cfg.update(th); cfg['k_F'] = kF
            else: cfg['q_min'] = 0.0
            cal_streams = [(P, run(P, experts, ref, dict(cfg, head=False, recal=False, tent=False))) for P in cal]
            theta_S, _ = source_theta(cal_streams)
            if theta_S is None: theta_S = float(THETAS[0])
            a, b = platt(cal_streams)
            rwin = np.concatenate([o['risk'][(P['y'] == 0) & ~np.isnan(o['risk'])] for P, o in cal_streams])
            cfg['theta_S'] = theta_S
            meta = dict(variant=v, train=tm, dataset=dataset, fold=fold, seed=seed, theta_S=theta_S,
                        theta_win=float(np.quantile(rwin, 1 - ALPHA)), platt=[a, b], t_train=t_train,
                        cfg={k_: (list(x) if isinstance(x, tuple) else x) for k_, x in cfg.items()})
            for P in tg:
                tt = time.time(); o = run(P, experts, ref, cfg); dt = time.time() - tt
                np.savez_compressed(out + tag + f"/{v}/{P['pid']}.npz", risk=o['risk'].astype(np.float32),
                    src_risk=o['src_risk'].astype(np.float32), theta=o['theta'].astype(np.float32), state=o['state'],
                    DF=o['DF'].astype(np.float16), DE=o['DE'].astype(np.float16), Q=o['Q'].astype(np.float16),
                    gamma=o['gamma'][::15].astype(np.float16), sec=dt, n_updates=o['n_updates'], recal=np.array(o['recal_log']))
            json.dump(meta, open(out + tag + f'/{v}/meta.json', 'w'), indent=1)
            print(tag, v, 'done', time.strftime('%H:%M'), flush=True)
        torch.save([e.state_dict() for e in experts], out + tag + f'/experts_{tm}.pt')

if __name__ == '__main__':
    ds, folds_, seeds = sys.argv[1], [int(x) for x in sys.argv[2].split(',')], [int(x) for x in sys.argv[3].split(',')]
    for s in seeds:
        for f in folds_:
            run_fold(ds, f, s)
