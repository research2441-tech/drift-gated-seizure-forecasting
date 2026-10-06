"""Expert training on source-patient inputs produced by the same causal procedure used at test."""
import numpy as np, torch, torch.nn.functional as F
from config import *
from prep import load, labels
from source_ref import SourceRef
from deploy import run, RF
from model import Expert

CHUNK = 96

def prepare(pids):
    return [labels(load(p)) for p in pids]

def source_inputs(pats, ref, cfg):
    """model-free causal pass over each source stream; returns list of input dicts."""
    out = []
    for P in pats:
        o = run(P, None, ref, cfg, keep_inputs=True)
        out.append(dict(inp=o['inp'], m=o['usable'].astype(np.float32), y=P['y'], seg=P['seg'], DF=o['DF'], state=o['state']))
    return out

def fit_kF(pats, ref, cfg, q=0.95):
    """feature-change threshold: q-quantile of D_F on source interictal windows (ungated pass)."""
    c = dict(cfg); c['gate'] = None
    v = []
    for P in pats:
        o = run(P, None, ref, c)
        sel = (P['y'] == 0) & (o['state'] > 0) & (o['DF'] > 0)
        v.append(o['DF'][sel])
    return float(np.quantile(np.concatenate(v), q))

def train_expert(data, seed=0, steps=1500, bs=16, lr=2e-3):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    for d in data:
        seg = d['seg']; n = len(seg)
        ok = np.zeros(n, bool); ok[CHUNK - 1:] = seg[CHUNK - 1:] == seg[:n - CHUNK + 1]
        d['pos'] = np.flatnonzero(ok & (d['y'] == 1)); d['neg'] = np.flatnonzero(ok & (d['y'] == 0))
    has_pos = [i for i, d in enumerate(data) if len(d['pos'])]
    has_neg = [i for i, d in enumerate(data) if len(d['neg'])]
    net = Expert(); opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    for it in range(steps):
        X, M, Y = [], [], []
        for b in range(bs):
            d = data[rng.choice(has_pos)] if b % 2 == 0 else data[rng.choice(has_neg)]
            e = rng.choice(d['pos'] if b % 2 == 0 else d['neg'])
            s0 = e - CHUNK + 1
            X.append(d['inp'][s0:e + 1]); M.append(d['m'][s0:e + 1]); Y.append(d['y'][s0 + RF:e + 1])
        X = torch.from_numpy(np.stack(X).astype(np.float32)); M = torch.from_numpy(np.stack(M))
        Y = torch.from_numpy(np.stack(Y).astype(np.float32))
        drop = (torch.rand(bs, 1, 18) < 0.10).float(); M2 = M * (1 - drop)
        X = (X + 0.05 * torch.randn_like(X)) * M2.unsqueeze(-1)
        logit = net(X, M2)
        wp = (Y == 1).float(); wn = (Y == 0).float()
        w = wp / wp.sum().clamp(min=1) + wn / wn.sum().clamp(min=1)
        loss = (F.binary_cross_entropy_with_logits(logit, Y.clamp(min=0), reduction='none') * w).sum() / 2
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    net.eval()
    return net

def train_ensemble(data, K=3, seed=0, frac=0.75, **kw):
    """K experts, each on a random 75% subset of source patients (patient bagging)."""
    rng = np.random.default_rng(1000 + seed); ex = []
    for k in range(K):
        idx = rng.permutation(len(data))[:max(2, int(round(frac * len(data))))]
        ex.append(train_expert([data[i] for i in idx], seed=seed * 10 + k, **kw))
    return ex
