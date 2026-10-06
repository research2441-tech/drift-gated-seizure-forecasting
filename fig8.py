import json, numpy as np
from fig_style import *
d = json.load(open('./results/summary_chb.json')); ds = {r[0]: r for r in json.load(open('./results/datasets.json'))}
C = json.load(open('./results/confound.json'))
D = d['main']['DRIFT']; P = d['pids']
fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.9), gridspec_kw=dict(wspace=0.5, left=0.13, right=0.98, bottom=0.24, top=0.8))
ax = axs[0]
for i, p in enumerate(P):
    h = max(ds[p][5], 0.3); tw = D['tiw_pp'][i]
    full = ds[p][5] >= 12
    ax.scatter(h, tw, s=13, marker='o', facecolor='black' if full else 'white', edgecolor='black', lw=0.6, zorder=3)
ax.axhline(0.2, ls='--', color='0.4', lw=0.7); ax.axvline(12, ls=':', color='0.5', lw=0.7)
ax.text(240, 0.21, '$\\alpha=0.2$', fontsize=6, color='0.25', ha='right')
ax.text(13.5, 0.27, '12 h', fontsize=6, color='0.25')
ax.set_xscale('log'); import matplotlib.ticker as mt; ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v,_: f'{v:g}')); ax.set_xlim(0.25, 250); ax.set_ylim(-0.015, 0.31)
ax.set_xlabel('Adjudicated interictal hours'); ax.set_ylabel('Deployed TiW')
ax.scatter([], [], s=13, facecolor='black', edgecolor='black', label='recalibrated')
ax.scatter([], [], s=13, facecolor='white', edgecolor='black', label='source only')
ax.legend(loc='lower center', bbox_to_anchor=(0.6, 1.02), ncol=2, handletextpad=0.2, columnspacing=0.8, fontsize=5.8)
ax = axs[1]
grp = [[v['auc'] for v in C.values() if v['complete']], [v['auc'] for v in C.values() if not v['complete']]]
for j, g in enumerate(grp):
    x = j + np.random.default_rng(j).uniform(-0.12, 0.12, len(g))
    ax.scatter(x, g, s=10, facecolor='0.6', edgecolor='black', lw=0.4, zorder=3)
    ax.plot([j - 0.22, j + 0.22], [np.median(g)] * 2, color='black', lw=1.3)
ax.axhline(0.5, ls=':', color='0.4', lw=0.7)
ax.set_xticks([0, 1]); ax.set_xticklabels([f'complete\n(n={len(grp[0])})', f'excerpts\n(n={len(grp[1])})'])
ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.03, 1.05); ax.set_ylabel('AUROC of time since\nsegment start')
panel(axs[0], '(a)'); panel(axs[1], '(b)')
fig.savefig('fig8_validity.png'); print('ok')
