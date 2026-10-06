import json, numpy as np, collections
from fig_style import *
d = json.load(open('./results/semisynth2.json'))
M = ['STATIC', 'ALIGN', 'SLOWALIGN', 'ALIGN+G', 'GAIN', 'DRIFT']
per = collections.defaultdict(lambda: collections.defaultdict(list))
for k, v in d.items():
    pid, seed, m, dl = k.split('|'); per[(m, float(dl))][pid].append(v)
def stat(m, dl, key):
    vals = [np.nanmean([x.get(key, np.nan) for x in L]) for L in per[(m, dl)].values()]
    vals = np.array(vals); rng = np.random.default_rng(0)
    bs = [np.nanmean(rng.choice(vals, len(vals))) for _ in range(2000)]
    return np.nanmean(vals), np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)
fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.45), gridspec_kw=dict(wspace=0.5, left=0.13, right=0.97, bottom=0.36, top=0.90))
deltas = [0.05, 0.1, 0.2]
for i, m in enumerate(M):
    fc, ec, hat, mk, ls = SYS[m]
    for ax, key in zip(axs, ['retained', 'auc']):
        mu = [stat(m, dl, key) for dl in deltas]
        x = np.arange(3) + (i - 2.5) * 0.11
        ax.errorbar(x, [u[0] for u in mu], yerr=[[u[0] - u[1] for u in mu], [u[2] - u[0] for u in mu]], fmt=mk,
                    ms=3.2 if mk != '*' else 5, mfc=fc, mec='black', color='black', lw=0.6, ls=ls, capsize=1.2, elinewidth=0.5, label=m)
for ax, yl in zip(axs, ['Retained contrast (z units)', 'AUROC (late vs control)']):
    ax.set_xticks(range(3)); ax.set_xticklabels(['0.05', '0.10', '0.20']); ax.set_xlabel('Injected change $\\delta$')
    ax.set_ylabel(yl)
axs[1].axhline(0.5, color='0.6', lw=0.5, ls=':')
panel(axs[0], '(a)'); panel(axs[1], '(b)')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.54, 0.0), handlelength=2.6, columnspacing=1.6, fontsize=6.2)
fig.savefig('fig3_semisynth.png')
print('ok')
