import json, numpy as np
from fig_style import *
from matplotlib.patches import Rectangle
d = json.load(open('./results/fig2_trace.json'))
t = np.array(d['t']) / 3600; H = 20.5; m = t <= H
def smooth(v, k=75):
    v = np.array(v, float); c = np.cumsum(np.r_[0, v]); o = np.empty_like(v)
    for i in range(len(v)): lo = max(0, i - k + 1); o[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return o
fig, axs = plt.subplots(3, 1, figsize=(3.5, 3.6), sharex=True, gridspec_kw=dict(height_ratios=[2.4, 1.1, 0.55], hspace=0.32, left=0.2, right=0.97, top=0.93, bottom=0.11))
ax = axs[0]
for o in d['onsets']:
    a, b = (o - 2100) / 3600, o / 3600
    if a < H:
        for x in axs: x.axvspan(a, b, color='0.88', lw=0)
names = {'gain': 'gain step', 'emg': 'EMG', 'loss': 'ch. loss'}
for j, (k, o) in enumerate(d['kinds']):
    a = o / 3600
    if a < H:
        b = a + 1; y0 = 2.55 if j % 2 == 0 else 1.75
        ax.annotate('', xy=(a, y0), xytext=(b, y0), arrowprops=dict(arrowstyle='<->', lw=0.6, color='0.25', shrinkA=0, shrinkB=0))
        ax.text((a + b) / 2, y0 + 0.17, names[k], ha='center', va='bottom', fontsize=5.6, color='0.15')
for name, ls, lw, col in [('ALIGN', (0, (4, 2)), 0.8, '0.45'), ('STATIC', (0, (1, 1)), 0.8, '0.3'), ('DRIFT', '-', 0.9, 'black')]:
    ax.plot(t[m], smooth(d[name]['score'])[m], ls=ls, lw=lw, color=col, label=name)
ax.set_ylabel('Shape-deviation\nscore (z)'); ax.set_ylim(-1.6, 3.2)
ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.17), ncol=3, fontsize=6, handlelength=2.4)
ax.text(d['onsets'][0] / 3600 - 0.29, -1.45, 'injected\nperiod', fontsize=5.6, ha='center', va='bottom', color='0.2')
ax2 = axs[1]
g_d = np.array(d['DRIFT']['gamma']); g_s = np.array(d['STATIC']['gamma'])
ax2.plot(t[m], g_d[m], color='black', lw=0.9, label='DRIFT gain $\\gamma_{t,1}$')
ax2.set_ylabel('log$_{10}$ gain'); ax2.set_ylim(-0.25, 0.55)
ax2.legend(loc='upper left', fontsize=6, handlelength=2)
ax3 = axs[2]
st = np.array(d['DRIFT']['state'])[m]; tt = t[m]
for val, shade, lab in [(1, '0.55', 'hold'), (0, '0.05', 'unavailable')]:
    sel = st == val
    ax3.vlines(tt[sel], 0, 1, color=shade, lw=0.35)
ax3.set_yticks([]); ax3.set_ylim(0, 1); ax3.set_ylabel('Gate', rotation=0, ha='right', va='center')
ax3.set_xlabel('Monitoring time (h)'); ax3.set_xlim(0, H)
from matplotlib.lines import Line2D
ax3.legend(handles=[Line2D([0], [0], color='0.55', lw=2), Line2D([0], [0], color='0.05', lw=2)], labels=['hold', 'unavailable'],
           loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=5.8, handlelength=1.2, borderaxespad=0.1)
for x, s in zip(axs, ['(a)', '(b)', '(c)']): x.text(-0.21, 1.03, s, transform=x.transAxes, fontsize=8, weight='bold', va='top')
fig.savefig('fig2_trace.png'); print('ok')
