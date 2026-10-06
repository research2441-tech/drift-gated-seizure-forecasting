"""Montage-aware window encoder + causal temporal network (one source expert)."""
import torch, torch.nn as nn, torch.nn.functional as F
NC, NF = 18, 33          # 16 absolute + 16 reference-relative + 1 relative gain

class MontageEncoder(nn.Module):
    """Shared per-channel map, channel identity embedding, availability-masked attention pooling."""
    def __init__(s, d=48, dz=64):
        super().__init__()
        s.phi = nn.Sequential(nn.Linear(NF, d), nn.GELU(), nn.Linear(d, d))
        s.emb = nn.Parameter(torch.zeros(NC, d))
        s.norm = nn.LayerNorm(d)
        s.att = nn.Linear(d, 1)
        s.out = nn.Linear(2 * d + 1, dz)
    def forward(s, x, m):
        # x [..., NC, NF], m [..., NC] (1 usable, 0 unavailable)
        h = s.norm(s.phi(x) + s.emb)
        a = s.att(torch.tanh(h)).squeeze(-1).masked_fill(m < 0.5, -1e4)
        a = torch.softmax(a, -1) * (m.sum(-1, keepdim=True) > 0)
        pooled = (a.unsqueeze(-1) * h).sum(-2)
        mean = (m.unsqueeze(-1) * h).sum(-2) / m.sum(-1, keepdim=True).clamp(min=1)
        frac = m.mean(-1, keepdim=True)
        return F.gelu(s.out(torch.cat([pooled, mean, frac], -1)))

class CausalTCN(nn.Module):
    def __init__(s, dz=64, dil=(1, 2, 4, 8)):
        super().__init__()
        s.dil = dil
        s.convs = nn.ModuleList([nn.Conv1d(dz, dz, 3, dilation=d) for d in dil])
        s.head = nn.Linear(dz, 1)
    RF = 30                       # receptive field minus one
    def hidden(s, z):             # z [B, L, dz] -> causal state [B, L - RF, dz]
        h = z.transpose(1, 2)
        for c, d in zip(s.convs, s.dil):
            h = h[:, :, 2 * d:] + F.gelu(c(h))     # valid causal convolution with residual
        return h.transpose(1, 2)
    def forward(s, z):            # per-step logits [B, L - RF]
        return s.head(s.hidden(z)).squeeze(-1)

class Expert(nn.Module):
    def __init__(s):
        super().__init__()
        s.enc, s.tcn = MontageEncoder(), CausalTCN()
    def forward(s, x, m):         # x [B, L, NC, NF], m [B, L, NC] -> logits [B, L-30]
        return s.tcn(s.enc(x, m))
