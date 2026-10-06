"""Controlled recording disturbances injected into real target streams, scored with trained experts.
Each disturbance occupies one hour of interictal time (>= 2 h from any seizure):
  gain  : +0.30 log10 amplitude (x2) on six channels, persisting to the end of the stream
  loss  : three channels flat (disconnected) for 1 h
  emg   : broadband muscle contamination on six channels for 1 h
Outcome: change in false alarms per hour and time in warning inside the disturbed hour,
and the fraction of the hour reported as unavailable or held."""
import numpy as np
from config import *

FOC = [0, 1, 2, 3, 4, 5]
def apply(P, kind, t0, dur=3600.0):
    Q = dict(P); x = P['x'].copy(); t = P['t']
    sel = (t > t0) & (t <= (t0 + dur if kind != 'gain' else t[-1] + 1)); idx = np.flatnonzero(sel)
    if kind == 'gain':
        x[np.ix_(idx, FOC)] += 0.30 * np.r_[[2.0] * 13, 1.0, 1.0, 0.0]
    elif kind == 'loss':
        ch = FOC[:3]
        x[np.ix_(idx, ch, np.arange(13))] = -9.0; x[np.ix_(idx, ch, [13])] = -6.0
        x[np.ix_(idx, ch, [14])] = -3.0; x[np.ix_(idx, ch, [15])] = 1.0
    elif kind == 'emg':
        emg = np.array([-2, -2, -2, -2, -1.5, -1, -0.5, 0, 0.3, 0.5, 0.7, 0.8, 1.0]) + 1.2
        x[np.ix_(idx, FOC, np.arange(13))] = np.log10(10 ** x[np.ix_(idx, FOC, np.arange(13))] + 10 ** emg)
        x[np.ix_(idx, FOC, [13])] += 0.25; x[np.ix_(idx, FOC, [14])] += 0.15
    Q['x'] = x
    return Q

def pick_times(P, n=2, rng=None):
    """start times of 1-h blocks inside interictal stretches, >= 2 h after stream start"""
    t = P['t']; inter = P['y'] == 0
    cands = []
    for i in range(0, len(t), 225):
        a = t[i]; b = a + 3600
        if a - t[0] < 7200: continue
        m = (t > a) & (t <= b)
        if m.sum() > 800 and inter[m].all() and len(np.unique(P['seg'][m])) == 1: cands.append(a)
    rng = rng or np.random.default_rng(0)
    if not cands: return []
    return list(rng.choice(cands, size=min(n, len(cands)), replace=False))
