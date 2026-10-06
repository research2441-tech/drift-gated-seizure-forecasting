"""Source-side preparation: static per-patient gain, source reference statistics, expert training."""
import numpy as np, torch, torch.nn.functional as F
from config import *
from prep import load, labels, channel_quality, GAIN_LOAD

def gain_estimate(x, mu, w):
    """Weighted least-squares log10 gain per window and channel (projection onto GAIN_LOAD)."""
    l = GAIN_LOAD
    return ((x - mu) * (w * l)).sum(-1) / (w * l * l).sum()

class SourceRef:
    """Population reference (mu_S per channel/feature, sigma_S per feature) after gain removal."""
    def fit(s, pats):
        xs = []
        for P in pats:
            sel = (P['y'] == 0)
            xs.append(P['x'][sel][::5])
        X = np.concatenate(xs)
        s.hf = np.median(X[..., 12] - X[..., 6:9].mean(-1), 0)   # per-channel high-frequency ratio reference
        s.mu = np.median(X, 0)                             # [NC, NF]
        s.sd = X.reshape(-1, X.shape[-1]).std(0) + 1e-3    # [NF]
        s.w = 1.0 / s.sd ** 2
        for _ in range(2):                                 # alternate gains and reference
            xs = []
            for P in pats:
                g = np.median(gain_estimate(P['x'][P['y'] == 0][::5], s.mu, s.w), 0)
                P['g_static'] = g.astype(np.float32)
                xs.append(P['x'][P['y'] == 0][::5] - g[:, None] * GAIN_LOAD)
            X = np.concatenate(xs)
            s.mu = np.median(X, 0); s.sd = X.reshape(-1, X.shape[-1]).std(0) + 1e-3; s.w = 1 / s.sd ** 2
        return s
    def norm(s, x, g):
        """x [..., NC, NF], g [..., NC] -> standardised input."""
        return (x - g[..., None] * GAIN_LOAD - s.mu) / s.sd

