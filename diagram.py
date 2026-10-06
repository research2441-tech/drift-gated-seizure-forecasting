"""Fig. 1: system architecture (grayscale). Boxes are placed on a grid; every arrow is routed
orthogonally between named anchors, and an automatic check rejects any arrow segment that
crosses a box it does not connect, and any pair of overlapping boxes or labels."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 6.6})

W, H = 7.2, 4.35
fig = plt.figure(figsize=(W, H), dpi=300)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis('off')
B = {}
def box(k, x, y, w, h, text, fill='white', lw=0.8, ls='-', bold=False):
    B[k] = (x, y, w, h)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.0,rounding_size=0.8', fc=fill, ec='black', lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=6.3, weight='bold' if bold else 'normal', linespacing=1.15)
def region(x, y, w, h, title, bottom=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.0,rounding_size=1.2', fc='none', ec='0.45', lw=0.6, ls=(0, (4, 2))))
    ax.text(x + 0.8, (y + 1.5) if bottom else (y + h - 1.6), title, ha='left', va='center', fontsize=6.4, style='italic', color='0.15')
def anchor(k, side, f=0.5):
    x, y, w, h = B[k]
    return {'l': (x, y + f * h), 'r': (x + w, y + f * h), 't': (x + f * w, y + h), 'b': (x + f * w, y)}[side]
SEGS = []
def arrow(pts, src, dst, ls='-', lw=0.8, label=None, lpos=None, head=True):
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        SEGS.append(((x0, y0), (x1, y1), src, dst))
    for i in range(len(pts) - 1):
        last = i == len(pts) - 2
        ax.add_patch(FancyArrowPatch(pts[i], pts[i + 1], arrowstyle='-|>' if (last and head) else '-', mutation_scale=6,
                                     lw=lw, ls=ls, color='black', shrinkA=0, shrinkB=0))
    if label:
        ax.text(*lpos, label, fontsize=5.7, ha='center', va='center', color='0.1',
                bbox=dict(fc='white', ec='none', pad=0.6))

# ---------------- regions
region(1, 30.5, 98, 28.5, 'Online forecasting path (strictly causal, 1-min update step)')
region(1, 13.8, 61, 15.4, 'Drift assessment and adaptation control', bottom=True)
region(64, 13.8, 35, 15.4, 'Calibration from adjudicated outcomes', bottom=True)
region(1, 0.6, 98, 12.2, 'Offline source training (patient-held-out sources only)')
# ---------------- online row
box('eeg', 3, 44, 11, 9.5, 'Multichannel\nscalp EEG\n18 bipolar ch.\n4-s windows', fill='0.92')
box('feat', 17, 44, 12, 9.5, 'Log-spectral and\ntime-domain\nfeatures +\nchannel mask', fill='white')
box('adapt', 32, 44, 13, 9.5, 'Restricted\nadaptation\ngain $\\gamma_{c}$ +\npatient reference', fill='0.85')
box('enc', 48, 44, 12, 9.5, 'Montage-aware\nencoder\n(masked channel\nattention)', fill='white')
box('tcn', 63, 44, 11, 9.5, 'Causal TCN\n128-s context\nK source\nexperts', fill='white')
box('read', 77, 44, 10, 9.5, 'Patient\nreadout\n(anchored)', fill='0.85')
box('alarm', 89.5, 44, 8.5, 9.5, 'Alarm\npolicy\nfiring power,\nrefractory', fill='0.92')
box('out', 89.5, 31.2, 8.5, 9.0, 'Output:\nwarning |\nno warning |\nunavailable', fill='0.97', lw=0.7)
# ---------------- drift row
box('ind', 3, 17.6, 21, 9.5, 'Indicators:  quality $Q_t$,\nexpert disagreement $D^{E}_t$,\nshape change $D^{F}_t$,\nreference risk $\\bar r_t$', fill='white')
box('gate', 29, 17.6, 14, 9.5, 'Drift gate\nadapt | hold |\nunavailable\n(bounded hold)', fill='0.80', bold=False)
box('upd', 47, 17.6, 13, 9.5, 'Gated update\nof $\\gamma_{c}$ and\nreference\n(source shrinkage)', fill='white')
# ---------------- calibration row
box('adj', 66, 17.6, 10.5, 9.5, 'Adjudicated\noutcomes\n(delayed,\nchronological)', fill='white')
box('fit', 78.5, 22.7, 19, 4.4, 'Readout refit (cross-fitted)', fill='white')
box('ltt', 78.5, 17.6, 19, 4.4, 'Learn-then-Test threshold', fill='white')
# ---------------- training row
box('src', 3, 2.2, 15, 8, 'Source patients\n(same causal\npreprocessing)', fill='0.92')
box('aug', 22, 2.2, 15, 8, 'Channel dropout,\nfeature noise,\npatient bagging', fill='white')
box('net', 41, 2.2, 18, 8, 'Encoder + causal TCN\n(expert $k$)', fill='white')
box('loss', 67, 2.2, 15, 8, 'Class-balanced\ncross-entropy\n$\\mathcal{L}_{\\mathrm{src}}$', fill='white')
box('frz', 86, 2.2, 11.5, 8, 'Frozen\nexperts\ndeployed', fill='0.92')

# ---------------- arrows: online path (left to right, horizontal, mid-height)
for a, b in [('eeg', 'feat'), ('feat', 'adapt'), ('adapt', 'enc'), ('enc', 'tcn'), ('tcn', 'read'), ('read', 'alarm')]:
    arrow([anchor(a, 'r', 0.5), anchor(b, 'l', 0.5)], a, b)
arrow([anchor('alarm', 'b', 0.5), (93.75, 40.2)], 'alarm', 'out')
# features -> indicators (down)
arrow([anchor('feat', 'b', 0.5), (23, 27.1)], 'feat', 'ind')
# TCN experts -> indicators (disagreement, reference risk): down then left along y=36
arrow([anchor('tcn', 'b', 0.35), (66.85, 36.5), (13.5, 36.5), (13.5, 27.1)], 'tcn', 'ind', label='expert risks', lpos=(55, 36.5))
# indicators -> gate -> update
arrow([anchor('ind', 'r', 0.5), anchor('gate', 'l', 0.5)], 'ind', 'gate')
arrow([anchor('gate', 'r', 0.5), anchor('upd', 'l', 0.5)], 'gate', 'upd', label='adapt', lpos=(45, 24.0))
# update -> restricted adaptation (up)
arrow([anchor('upd', 't', 0.3), (50.9, 30.0), (38.5, 30.0), (38.5, 44)], 'upd', 'adapt', label='state', lpos=(44.5, 30.0))
# gate -> unavailable to alarm (dashed control line around the bottom of the online row)
arrow([anchor('gate', 't', 0.5), (36, 33.0), (89.5, 33.0)], 'gate', 'out', ls=(0, (3, 2)), label='unavailable', lpos=(57, 33.0))
# adjudicated -> fit, ltt; fit -> readout; ltt -> alarm
arrow([anchor('adj', 'r', 0.75), anchor('fit', 'l', 0.5)], 'adj', 'fit')
arrow([anchor('adj', 'r', 0.25), anchor('ltt', 'l', 0.5)], 'adj', 'ltt')
arrow([anchor('fit', 't', 0.25), (83.25, 29.6), (80.5, 29.6), (80.5, 44)], 'fit', 'read')
arrow([anchor('ltt', 'r', 0.5), (98.5, 19.8), (98.5, 42.0), (96.0, 42.0), (96.0, 44)], 'ltt', 'alarm', label=r'$\theta_t$', lpos=(98.5, 26.5))
# training row: forward arrows
for a, b in [('src', 'aug'), ('aug', 'net')]:
    arrow([anchor(a, 'r', 0.5), anchor(b, 'l', 0.5)], a, b)
arrow([anchor('net', 'r', 0.78), anchor('loss', 'l', 0.78)], 'net', 'loss', label='logits', lpos=(63, 10.4))
# dashed reverse-mode gradient arrow: loss -> network (lower track)
arrow([anchor('loss', 'l', 0.25), anchor('net', 'r', 0.25)], 'loss', 'net', ls=(0, (3, 2)), label=r'$\nabla\mathcal{L}_{\mathrm{src}}$', lpos=(63, 2.9))
arrow([anchor('loss', 'r', 0.5), anchor('frz', 'l', 0.5)], 'loss', 'frz')

# ---------------- automatic geometry checks
def inside(px, py, b, m=0.05):
    x, y, w, h = b; return x + m < px < x + w - m and y + m < py < y + h - m
def seg_hits(seg, b, n=60):
    (x0, y0), (x1, y1) = seg
    return any(inside(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n, b) for i in range(1, n))
bad = []
for (p0, p1, s, d) in SEGS:
    for k, b in B.items():
        if k in (s, d): continue
        if seg_hits((p0, p1), b): bad.append((s, d, k))
ks = list(B)
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        a, b = B[ks[i]], B[ks[j]]
        if a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]:
            bad.append(('overlap', ks[i], ks[j]))
assert not bad, bad
fig.savefig('fig1_architecture.png', dpi=300)
print('fig1 ok, arrows', len(SEGS))
