"""Alarm policy (firing power + persistence + refractory warning), event-level metrics,
and Learn-then-Test threshold calibration for the complete policy."""
import numpy as np
from math import log, exp
from scipy.stats import binom
from config import *

def firing_power(risk, theta, seg, L=None):
    """fraction of available windows in the last L of the same segment with risk >= theta,
    and the number of available windows. risk NaN = unavailable."""
    L = L or FP_WIN
    av = ~np.isnan(risk)
    f = (np.nan_to_num(risk, nan=-1.0) >= theta).astype(np.float64)
    fp = np.zeros(len(risk)); cnt = np.zeros(len(risk))
    b = np.r_[0, np.flatnonzero(np.diff(seg)) + 1, len(seg)]
    for s, e in zip(b[:-1], b[1:]):
        cf = np.r_[0, np.cumsum(f[s:e])]; ca = np.r_[0, np.cumsum(av[s:e])]
        idx = np.arange(1, e - s + 1); lo = np.maximum(idx - L, 0)
        fp[s:e] = cf[idx] - cf[lo]; cnt[s:e] = ca[idx] - ca[lo]
    return fp / np.maximum(cnt, 1), cnt

def alarms_from_candidates(t, cand):
    """apply the refractory (warning) period to candidate alarm windows."""
    out, last = [], -np.inf
    for i in np.flatnonzero(cand):
        if t[i] >= last + WARN:
            out.append(t[i]); last = t[i]
    return np.array(out)

def policy(risk, t, seg, theta):
    fp, cnt = firing_power(risk, theta, seg)
    return alarms_from_candidates(t, (fp >= FP_FRAC) & (cnt >= FP_WIN / 2))

def policy_online(risk, t, seg, thetas_t):
    """thetas_t: per-window threshold in force at that time (piecewise constant)."""
    cand = np.zeros(len(t), bool)
    for th in np.unique(thetas_t):
        fp, cnt = firing_power(risk, th, seg)
        sel = thetas_t == th
        cand[sel] = ((fp >= FP_FRAC) & (cnt >= FP_WIN / 2))[sel]
    return alarms_from_candidates(t, cand)

def interictal_mask(t, on, off):
    m = np.ones(len(t), bool)
    for o, f in zip(on, off):
        m[(t > o - WARN) & (t <= f + POST_GAP)] = False
    return m

def evaluable(t, on, lead):
    """lead seizures whose occurrence interval is at least half recorded"""
    ev = np.zeros(len(on), bool)
    for i, o in enumerate(on):
        if lead[i]:
            k = ((t > o - SPH - SOP) & (t <= o - SPH)).sum()
            ev[i] = k * WSEC >= 0.5 * SOP
    return ev

def warning_cover(t, alarms):
    w = np.zeros(len(t), bool)
    for a in alarms: w[(t > a) & (t <= a + WARN)] = True
    return w

def metrics(alarms, t, on, off, lead, risk=None):
    ev = evaluable(t, on, lead)
    inter = interictal_mask(t, on, off)
    warn = warning_cover(t, alarms)
    hit, lt = [], []
    for o in on[ev]:
        ok = alarms[(alarms + SPH <= o) & (alarms + SPH + SOP >= o)]
        hit.append(len(ok) > 0)
        if len(ok): lt.append((o - ok.min()) / 60.0)
    true_alarm = np.array([np.any((on >= a + SPH) & (on <= a + SPH + SOP)) for a in alarms], bool) if len(alarms) else np.zeros(0, bool)
    inter_h = inter.sum() * WSEC / 3600
    fa_inter = sum(1 for a, ta in zip(alarms, true_alarm) if not ta)
    res = dict(n_events=int(ev.sum()), sens=float(np.mean(hit)) if hit else np.nan,
               fa_h=fa_inter / max(inter_h, 1e-9), tiw=float(warn[inter].mean()) if inter.any() else np.nan,
               lead_min=float(np.mean(lt)) if lt else np.nan, n_alarms=len(alarms), inter_h=inter_h,
               hits=[bool(h) for h in hit])
    if risk is not None:
        res['coverage'] = float((~np.isnan(risk)).mean())
    return res

# ---------------- Learn-then-Test over the complete policy ----------------
def hb_pvalue(loss_mean, n, alpha):
    """Hoeffding-Bentkus p-value for H0: E[loss] > alpha (losses in [0,1])."""
    if n == 0: return 1.0
    a = min(loss_mean, alpha)
    h = 0.0 if a <= 0 else a * log(a / alpha)
    h += (1 - a) * log((1 - a) / (1 - alpha))
    p_h = exp(-n * h)
    p_b = exp(1) * binom.cdf(np.ceil(n * loss_mean), n, alpha)
    return min(p_h, p_b, 1.0)

THETAS = np.round(np.r_[np.linspace(0.995, 0.90, 20), np.linspace(0.89, 0.30, 60)], 4)

def block_losses(t, inter, warn, block=3600.0):
    """per-hour-block interictal time-in-warning; blocks need >= 30 min of interictal time."""
    bid = np.floor(t / block).astype(np.int64)
    out = []
    for b in np.unique(bid[inter]):
        sel = (bid == b) & inter
        if sel.sum() * WSEC >= 1800: out.append((b, warn[sel].mean()))
    return out

def ltt_threshold(loss_by_theta, alpha=ALPHA, delta=0.10):
    """fixed-sequence test from the most conservative threshold downwards.
    loss_by_theta: list over THETAS (descending) of arrays of block losses."""
    chosen = None
    for th, L in zip(THETAS, loss_by_theta):
        if hb_pvalue(float(np.mean(L)) if len(L) else 1.0, len(L), alpha) <= delta: chosen = th
        else: break
    return chosen
