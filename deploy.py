"""Online, strictly causal processing of one monitoring stream.

Every 1-min step:
  1. the unsupervised adaptation state (per-channel log gain gamma, patient reference mu_P, v_P)
     maps incoming windows to an absolute input and a reference-relative input;
  2. the frozen source experts produce causal states h; source heads give the reference risk
     and the expert disagreement used by the drift assessment;
  3. the drift assessment labels each window 2 = adapt, 1 = hold, 0 = unavailable;
  4. gamma and the patient reference are updated from 'adapt' windows only.
Every hour (stream time) the patient readouts are refitted on adjudicated outcomes only,
anchored to the source readout and cross-fitted by hour parity; every few hours the alarm
threshold of the complete policy is recalibrated by Learn-then-Test on adjudicated
interictal hours using out-of-fold risks.
With experts=None the same pass runs model-free (quality + feature-change gate); this is
how source training inputs are produced, so training and deployment share one procedure.
"""
import numpy as np, torch, torch.nn.functional as F
from config import *
from prep import channel_quality, GAIN_LOAD
from alarm import THETAS, firing_power, alarms_from_candidates, warning_cover, interictal_mask, block_losses, ltt_threshold
RF = 30
STEP = 15
DEFAULT = dict(mode='gain', gate=('E', 'F', 'R'), use_quality=True, tau_a=1800.0, kappa=225.0,
               tau_ref=7200.0, short=75, warm=225, q_min=0.5, k_F=None, k_E=None, k_R=None, k_U=None,
               h_max=SPH + SOP, rel=True, freeze_after=None, recentre=False,
               head=True, lam=0.05, adj_delay=3600.0, upd_every=3600.0, min_pos=75,
               recal=True, recal_every=3 * 3600.0, recal_hist=48 * 3600.0, n_min=12, alpha=ALPHA,
               delta=0.10, theta_S=None, tent=False, tent_lr=1e-3, slow=False, tau_slow=600.0)

def seg_bounds(seg):
    b = np.r_[0, np.flatnonzero(np.diff(seg)) + 1, len(seg)]
    return list(zip(b[:-1], b[1:]))

def short_means(v, w, seg, L):
    out = np.zeros_like(v, dtype=np.float64)
    for s, e in seg_bounds(seg):
        cv = np.cumsum(np.concatenate([np.zeros((1,) + v.shape[1:]), v[s:e] * w[s:e][..., None]]), 0)
        cw = np.cumsum(np.concatenate([np.zeros((1,) + w.shape[1:]), w[s:e]]), 0)
        idx = np.arange(1, e - s + 1); lo = np.maximum(idx - L, 0)
        out[s:e] = (cv[idx] - cv[lo]) / np.maximum(cw[idx] - cw[lo], 1)[..., None]
    return out

def fit_head(H, y, w0, b0, lam, iters=30):
    """anchored, class-balanced logistic readout: BCE + lam * ||w - w0||^2"""
    Ht = torch.from_numpy(H.astype(np.float32)); yt = torch.from_numpy(y.astype(np.float32))
    w = torch.tensor(w0, requires_grad=True); b = torch.tensor(b0, requires_grad=True)
    pw = 0.5 / max(y.mean(), 1e-3); nw = 0.5 / max(1 - y.mean(), 1e-3)
    wt = torch.where(yt > 0.5, torch.tensor(pw), torch.tensor(nw))
    w0t = torch.tensor(w0)
    opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=iters, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss = (F.binary_cross_entropy_with_logits(Ht @ w + b, yt, reduction='none') * wt).mean() + lam * ((w - w0t) ** 2).sum()
        loss.backward(); return loss
    opt.step(closure)
    return w.detach().numpy().copy(), b.detach().numpy().copy()

def preictal_onset_index(P):
    """for each preictal-labelled window, the onset time that defines its label"""
    t = P['t']; on = P['onset']; lead = P['lead']
    o = np.full(len(t), np.inf)
    for x, l in zip(on, lead):
        if l: o[(t > x - SPH - SOP) & (t <= x - SPH)] = x
    return o

def policy_losses(risk, P, lo, hi, now, inter):
    """per-threshold hour-block interictal time-in-warning on adjudicated blocks in [lo, hi)"""
    t = P['t'][lo:hi]; seg = P['seg'][lo:hi]; r = risk[lo:hi]; it = inter[lo:hi]
    bid = np.floor(t / 3600).astype(np.int64)
    adj = (bid + 1) * 3600 + WARN <= now
    out = []
    for th in THETAS:
        fp, cnt = firing_power(r, th, seg)
        al = alarms_from_candidates(t, (fp >= FP_FRAC) & (cnt >= FP_WIN / 2))
        warn = warning_cover(t, al)
        out.append(np.array([l for b, l in block_losses(t, it & adj, warn)]))
    return out

@torch.no_grad()
def _enc(e, x, m): return e.enc(x, m).numpy()

def run(P, experts, ref, cfg=None, keep_inputs=False):
    c = dict(DEFAULT); c.update(cfg or {})
    if experts is None: c['gate'] = tuple(g for g in (c['gate'] or ()) if g == 'F') or None
    x, miss, seg, t = P['x'], P['miss'], P['seg'], P['t']
    n, NC, NF = x.shape
    usable = channel_quality(x, miss, ref.hf) > 0.5 if c['use_quality'] else ~miss
    Q = usable.mean(1)
    rho_a = np.exp(-WSEC / c['tau_a']); rho_r = np.exp(-WSEC / c['tau_ref'])
    l = GAIN_LOAD.astype(np.float64); aw = (ref.w * l) / (ref.w * l * l).sum()
    dev = x - ref.mu
    ghat = (dev * aw).sum(-1)                                   # instantaneous log gain [n, NC]
    shape = dev - ghat[..., None] * l                           # gain-orthogonal residual
    S = short_means(shape, usable.astype(np.float64), seg, c['short'])
    num = np.zeros(NC); den = np.zeros(NC)
    zm = np.zeros((NC, NF)); zv = np.zeros((NC, NF)); zw = np.zeros(NC)
    pm = np.zeros((NC, NF)); pv = np.ones((NC, NF)); pw = 0.0
    sm = np.zeros((NC, NF)); sv = np.ones((NC, NF))
    K = len(experts) if experts else 0
    if K and c['tent']:
        import copy
        experts = [copy.deepcopy(e) for e in experts]
        tparams = [p_ for e in experts for p_ in e.enc.norm.parameters()]
        topt = torch.optim.SGD(tparams, lr=c['tent_lr'], momentum=0.9)
    Z = [np.zeros((n, 64), np.float32) for _ in range(K)]
    H = [np.zeros((n, 64), np.float32) for _ in range(K)]
    DR = 128 if c['slow'] else 64
    HR = [np.zeros((n, DR), np.float32) for _ in range(K)] if c['slow'] else H   # readout features
    rho_s = np.exp(-WSEC / c['tau_slow'])
    has = np.zeros(n, bool)
    src_logit = np.full((n, max(K, 1)), np.nan, np.float32)
    dep_logit = np.full((n, max(K, 1)), np.nan, np.float32)
    if K:
        W0 = [e.tcn.head.weight.detach().numpy()[0].astype(np.float32) for e in experts]
        B0 = [e.tcn.head.bias.detach().numpy().astype(np.float32) for e in experts]
        W0x = [np.r_[W0[k], np.zeros(DR - 64, np.float32)] for k in range(K)]
        heads = {par: [(W0x[k].copy(), B0[k].copy()) for k in range(K)] for par in (0, 1)}
        y = P['y']; ponset = preictal_onset_index(P); inter = interictal_mask(t, P['onset'], P['offset'])
        parity = (np.floor(t / 3600).astype(np.int64) % 2)
    theta = np.full(n, c['theta_S'] if c['theta_S'] is not None else 0.5)
    th_cur = theta[0]; next_upd = t[0] + c['upd_every']; next_cal = t[0] + c['recal_every']
    n_updates = 0; recal_log = []
    state = np.zeros(n, np.int8); DF = np.zeros(n); gam = np.zeros((n, NC), np.float32)
    INP = np.zeros((n, NC, 33), np.float16) if keep_inputs else None
    accepted = 0; hold_since = None; seen = 0
    for s_start, s_end in seg_bounds(seg):
        for i0 in range(s_start, s_end, STEP):
            i1 = min(s_end, i0 + STEP)
            xi = x[i0:i1]
            if c['mode'] in ('gain', 'fixed', 'none'):
                g = num / (c['kappa'] + den) if c['mode'] == 'gain' else (P['g_fixed'] if c['mode'] == 'fixed' else np.zeros(NC))
                u = xi - g[:, None] * GAIN_LOAD - ref.mu
                # optional slow re-centring of spectral shape on the gated patient reference
                absn = (u - sm) / ref.sd if (c['recentre'] and pw > 0) else u / ref.sd
            else:                                                # 'zscore': unrestricted per-feature alignment
                lam_ = (zw / (c['kappa'] + zw))[:, None]; w_ = np.maximum(zw, 1e-9)[:, None]
                mu = lam_ * zm / w_ + (1 - lam_) * ref.mu
                var = lam_ * np.maximum(zv / w_ - (zm / w_) ** 2, 1e-4) + (1 - lam_) * ref.sd ** 2
                absn = (xi - mu) / np.sqrt(var); g = np.zeros(NC); u = absn * ref.sd
            gam[i0:i1] = g
            if not (pw > 0 and c['rel']):
                rel = np.zeros_like(u)
            elif c['mode'] == 'zscore':
                rel = (u - pm) / np.sqrt(pv)                     # unrestricted: full feature deviation
            else:
                rel = (shape[i0:i1] - sm) / np.sqrt(sv)          # restricted: gain-orthogonal shape deviation
            relg = (ghat[i0:i1] - g)[..., None] / 0.1 if c['rel'] else np.zeros(u.shape[:2] + (1,))
            inp = np.concatenate([absn, np.clip(rel, -8, 8), np.clip(relg, -8, 8)], -1)
            m = usable[i0:i1].astype(np.float32)
            inp = (inp * m[..., None]).astype(np.float32)
            if keep_inputs: INP[i0:i1] = inp
            if K:
                xt = torch.from_numpy(inp); mt = torch.from_numpy(m)
                if c['tent'] and i1 - max(s_start, i0 - RF) > RF:
                    # Tent: one entropy-minimisation step on the encoder normalisation affine parameters,
                    # using the current chunk and its causal context (no labels, never suspended)
                    lo_ = max(s_start, i0 - RF)
                    xc = torch.from_numpy(Z[0][:0])  # placeholder to keep types simple
                    topt.zero_grad(); ent = 0.0
                    for k, e in enumerate(experts):
                        zc = torch.cat([torch.from_numpy(Z[k][lo_:i0]), e.enc(xt, mt)], 0)
                        lg_ = e.tcn(zc[None])[0]
                        pr = torch.sigmoid(lg_)
                        ent = ent + (-(pr * torch.log(pr + 1e-6) + (1 - pr) * torch.log(1 - pr + 1e-6))).mean()
                    ent.backward(); topt.step()
                for k, e in enumerate(experts):
                    Z[k][i0:i1] = _enc(e, xt, mt)
                lo = max(s_start, i0 - RF); first = max(s_start + RF, i0)
                if first < i1:
                    has[first:i1] = True
                    for k, e in enumerate(experts):
                        with torch.no_grad():
                            hh = e.tcn.hidden(torch.from_numpy(Z[k][lo:i1])[None])[0].numpy()
                        H[k][first:i1] = hh[-(i1 - first):]
                        src_logit[first:i1, k] = H[k][first:i1] @ W0[k] + B0[k][0]
                        if c['slow']:
                            # slow summary of the causal state (EMA, reset at segment start) for the readout
                            HR[k][first:i1, :64] = H[k][first:i1]
                            prev = HR[k][first - 1, 64:] if first - 1 >= s_start + RF else H[k][first]
                            for j in range(first, i1):
                                prev = rho_s * prev + (1 - rho_s) * H[k][j]; HR[k][j, 64:] = prev
                        par = parity[first:i1]
                        la = HR[k][first:i1] @ heads[0][k][0] + heads[0][k][1][0]
                        lb = HR[k][first:i1] @ heads[1][k][0] + heads[1][k][1][0]
                        dep_logit[first:i1, k] = 0.5 * (la + lb)
            # ---- drift assessment (source-reference outputs only)
            lg = src_logit[i0:i1]
            risk = (1 / (1 + np.exp(-lg))).mean(1) if K else np.zeros(i1 - i0)
            DE = lg.std(1) if K else np.zeros(i1 - i0)
            okc = usable[i0:i1] & (pw > 0)
            d2 = np.where(okc, ((S[i0:i1] - sm) ** 2 / sv).mean(-1), np.nan)     # per-channel shape change
            with np.errstate(all='ignore'):
                DF[i0:i1] = np.where(okc.sum(1) >= 2, np.nanquantile(np.where(okc, d2, np.nan), 0.9, axis=1), 0.0)
            avail = Q[i0:i1] >= c['q_min']
            if K: avail &= ~np.isnan(risk)
            if c['k_U'] is not None: avail &= ~(DE > c['k_U'])
            st = np.where(avail, 2, 0).astype(np.int8)
            if c['gate']:
                hold = np.zeros(i1 - i0, bool)
                if 'E' in c['gate'] and c['k_E'] is not None: hold |= DE > c['k_E']
                if 'R' in c['gate'] and c['k_R'] is not None: hold |= risk > c['k_R']
                if 'F' in c['gate'] and c['k_F'] is not None and accepted >= c['warm']:
                    hold |= DF[i0:i1] > c['k_F']
                # a suspension cannot outlast the occurrence period: a change that persists
                # longer than h_max is re-admitted as the new baseline
                if hold.any():
                    if hold_since is None: hold_since = t[i0]
                    if c['h_max'] is not None and t[i1 - 1] - hold_since > c['h_max']: hold[:] = False
                else:
                    hold_since = None
                st[(st == 2) & hold] = 1
            if c['freeze_after'] is not None and seen >= c['freeze_after']:
                st[st == 2] = 1
            seen += int((st > 0).sum())
            state[i0:i1] = st
            theta[i0:i1] = th_cur
            # ---- unsupervised updates from accepted windows only
            dec = rho_a ** (i1 - i0)
            num *= dec; den *= dec; zm *= dec; zv *= dec; zw *= dec
            acc = np.flatnonzero(st == 2)
            if len(acc):
                accepted += len(acc)
                uu = usable[i0:i1][acc]; w3 = uu[..., None]
                cnt = np.maximum(uu.sum(0), 1)[:, None]
                if c['mode'] == 'gain':
                    num += (ghat[i0:i1][acc] * uu).sum(0); den += uu.sum(0)
                elif c['mode'] == 'zscore':
                    zm += (xi[acc] * w3).sum(0); zv += (xi[acc] ** 2 * w3).sum(0); zw += uu.sum(0)
                ua = u[acc]; sa = shape[i0:i1][acc]
                pw_new = pw * rho_r ** (i1 - i0) + len(acc); lamr = len(acc) / pw_new
                mu_u = (ua * w3).sum(0) / cnt; mu_s = (sa * w3).sum(0) / cnt
                if pw == 0:
                    pm, sm = mu_u, mu_s
                    pv = np.maximum(((ua - pm) ** 2 * w3).sum(0) / cnt, 1e-2)
                    sv = np.maximum(((sa - sm) ** 2 * w3).sum(0) / cnt, 1e-3)
                else:
                    pm = pm + lamr * (mu_u - pm); sm = sm + lamr * (mu_s - sm)
                    pv = np.maximum((1 - lamr) * pv + lamr * ((ua - pm) ** 2 * w3).sum(0) / cnt, 1e-2)
                    sv = np.maximum((1 - lamr) * sv + lamr * ((sa - sm) ** 2 * w3).sum(0) / cnt, 1e-3)
                pw = pw_new
            if not K: continue
            now = t[i1 - 1]
            # ---- supervised readout update from adjudicated outcomes
            if c['head'] and now >= next_upd:
                next_upd = now + c['upd_every']
                done = np.arange(n) < i1
                known = done & has & (state > 0)
                pos = known & (y == 1) & (ponset <= now - c['adj_delay'])
                neg = known & (y == 0) & (t <= now - INTER_GAP)
                if pos.sum() >= c['min_pos']:
                    rng = np.random.default_rng(int(now))
                    for par in (0, 1):
                        ip = np.flatnonzero(pos & (parity == par)); ineg = np.flatnonzero(neg & (parity == par))
                        if len(ip) < c['min_pos'] // 2 or len(ineg) < 100: continue
                        if len(ineg) > 6000: ineg = rng.choice(ineg, 6000, replace=False)
                        if len(ip) > 3000: ip = rng.choice(ip, 3000, replace=False)
                        idx = np.r_[ip, ineg]; yy = np.r_[np.ones(len(ip)), np.zeros(len(ineg))]
                        heads[par] = [fit_head(HR[k][idx], yy, W0x[k], B0[k], c['lam']) for k in range(K)]
                    n_updates += 1
            # ---- threshold recalibration of the complete policy on adjudicated interictal hours
            if c['recal'] and now >= next_cal:
                next_cal = now + c['recal_every']
                lo_i = int(np.searchsorted(t, now - c['recal_hist']))
                oof = np.full((i1 - lo_i, K), np.nan)
                for k in range(K):
                    hk = HR[k][lo_i:i1]; pk = parity[lo_i:i1]
                    oa = hk @ heads[1][k][0] + heads[1][k][1][0]      # even hours scored by the odd-hour head
                    ob = hk @ heads[0][k][0] + heads[0][k][1][0]
                    oof[:, k] = np.where(pk == 0, oa, ob)
                r_oof = np.full(n, np.nan)
                r_oof[lo_i:i1] = (1 / (1 + np.exp(-oof))).mean(1)
                r_oof[lo_i:i1][~has[lo_i:i1] | (state[lo_i:i1] == 0)] = np.nan
                L = policy_losses(r_oof, P, lo_i, i1, now, inter)
                nb = len(L[0])
                if nb >= c['n_min']:
                    th = ltt_threshold(L, c['alpha'], c['delta'])
                    th_cur = th if th is not None else THETAS[0]
                recal_log.append((float(now), nb, float(th_cur)))
    out = dict(DF=DF, Q=Q, state=state, gamma=gam, usable=usable)
    if K:
        risk = (1 / (1 + np.exp(-dep_logit))).mean(1); risk[(state == 0) | ~has] = np.nan
        srisk = (1 / (1 + np.exp(-src_logit))).mean(1); srisk[(state == 0) | ~has] = np.nan
        out.update(risk=risk, src_risk=srisk, DE=np.nanstd(src_logit, 1), theta=theta,
                   n_updates=n_updates, recal_log=recal_log)
    if keep_inputs: out['inp'] = INP
    return out
