"""Decoding, labels and recording-quality indicators for a patient stream."""
import numpy as np
from config import *

GAIN_LOAD = np.array([2.0] * 13 + [1.0, 1.0, 0.0], np.float32)   # d(feature)/d(log10 gain)

def load(pid):
    d = np.load(DATA + pid + '.npz')
    a = d['X']
    x = a.astype(np.float32)
    x[..., :15] /= 1000.0; x[..., 15] /= 10000.0
    miss = a[..., 0] == -32768
    x[miss] = 0.0
    return dict(pid=pid, x=x, miss=miss, t=d['t'], seg=d['seg'], onset=d['onset'], offset=d['offset'])

def lead_seizures(on, off):
    lead = np.ones(len(on), bool)
    for i in range(1, len(on)):
        if on[i] - off[:i].max() < LEAD_GAP: lead[i] = False
    return lead

def labels(P):
    """y: 1 preictal, 0 interictal, -1 excluded (training labels only)."""
    t, on, off = P['t'], P['onset'], P['offset']
    lead = lead_seizures(on, off)
    y = np.zeros(len(t), np.int8)
    for o, f in zip(on, off):
        y[(t > o - INTER_GAP) & (t <= f + POST_GAP)] = -1
    for o, l in zip(on, lead):
        if l: y[(t > o - SPH - SOP) & (t <= o - SPH)] = 1
    P['y'], P['lead'] = y, lead
    return P

def channel_quality(x, miss, ref_hf):
    """Soft per-channel quality in [0,1] from flat-line, amplitude and EMG indicators.
    ref_hf: source median of the 65-95 Hz to 13-25 Hz log-PSD ratio (per channel)."""
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    flat = sig((0.30 - x[..., 15]) / 0.05) * sig((x[..., 14] - 0.30) / 0.10)   # not flat, not dead
    amp = sig((3.0 - x[..., 14]) / 0.10)                                        # max |x| below 1 mV
    hf = x[..., 12] - x[..., 6:9].mean(-1)                                      # 65-95 Hz relative to 13-25 Hz
    emg = sig((1.0 - (hf - ref_hf)) / 0.15)                                     # gain-invariant EMG indicator
    q = flat * amp * emg
    q[miss] = 0.0
    return q

def segments_context(seg, L=SEQ):
    """index matrix [n, L] of causal context, left-padded within each segment."""
    n = len(seg); idx = np.arange(n)
    start = np.zeros(n, np.int64)
    b = np.r_[0, np.flatnonzero(np.diff(seg)) + 1]
    for i, s in enumerate(b):
        e = b[i + 1] if i + 1 < len(b) else n
        start[s:e] = s
    off = np.arange(-L + 1, 1)
    M = idx[:, None] + off[None, :]
    return np.maximum(M, start[:, None])
