import json, numpy as np, sys
from fig_style import *
S = json.load(open('./results/summary_chb.json'))
ORDER = ['SRC', 'HEAD', 'TENT', 'STATIC', 'ALIGN', 'SLOWALIGN', 'ALIGN+G', 'GAIN', 'DRIFT']
ORDER = [v for v in ORDER if v in S['main']]
LAB = {}

def fig4():
    """per-patient matched-burden sensitivity and AUROC, paired to DRIFT"""
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.35), gridspec_kw=dict(wspace=0.42, left=0.12, right=0.98, bottom=0.27, top=0.9))
    for ax, m, yl in zip(axs, ['auc', 's20'], ['Window AUROC', 'Sen @ TiW$\\leq$0.2']):
        for i, v in enumerate(ORDER):
            y = np.array(S['main'][v][m + '_pp'], float)
            x = i + np.random.default_rng(i).uniform(-0.18, 0.18, len(y))
            fc, ec, hat, mk, ls = SYS.get(v, ('white', 'black', '', 'o', '-'))
            ax.scatter(x, y, s=5, facecolor='0.6', edgecolor='none', zorder=1)
            mu = np.nanmean(y); lo, hi = np.nanpercentile([np.nanmean(np.random.default_rng(k).choice(y[~np.isnan(y)], (~np.isnan(y)).sum())) for k in range(1000)], [2.5, 97.5])
            ax.errorbar(i, mu, yerr=[[mu - lo], [hi - mu]], fmt=mk, ms=4.5 if mk != '*' else 7, mfc=fc, mec='black', color='black', capsize=1.5, lw=0.7, zorder=3)
        ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([LAB.get(v, v) for v in ORDER], rotation=90, fontsize=5.6)
        ax.set_ylabel(yl)
        if m == 'auc': ax.axhline(0.5, color='0.5', lw=0.5, ls=':')
    panel(axs[0], '(a)'); panel(axs[1], '(b)')
    fig.savefig('fig4_main.png')

def fig5():
    """calibration validity on long recordings: realised interictal TiW per policy"""
    pols = [('fixed05', 'Fixed\n0.5'), ('winq', 'Window\nquantile'), ('ltt_src', 'LTT\nsource'), ('ltt_online', 'LTT\nadjudicated')]
    fig, ax = plt.subplots(figsize=(3.5, 2.0), gridspec_kw=dict(left=0.13, right=0.98, bottom=0.22, top=0.9))
    for j, (p, lab) in enumerate(pols):
        for k, v in enumerate(['SRC', 'ALIGN', 'DRIFT']):
            if v not in S['calib']: continue
            y = np.array(S['calib'][v][p]['tiw_pp'], float)
            x = j * 4 + k + np.random.default_rng(k).uniform(-0.15, 0.15, len(y))
            fc, ec, hat, mk, ls = SYS[v]
            ax.scatter(x, y, s=12, marker=mk, facecolor=fc, edgecolor='black', lw=0.5, zorder=3, label=v if j == 0 else None)
            ax.plot([j * 4 + k - 0.35, j * 4 + k + 0.35], [np.median(y)] * 2, color='black', lw=1.0)
    ax.axhline(0.2, color='0.3', lw=0.7, ls='--'); ax.text(15.6, 0.215, '$\\alpha$', fontsize=7, ha='right')
    ax.set_xticks([1, 5, 9, 13]); ax.set_xticklabels([l for _, l in pols], fontsize=6)
    ax.set_ylabel('Realised interictal TiW'); ax.set_xlim(-0.8, 15.8)
    ax.legend(loc='upper right', ncol=3, fontsize=6, handletextpad=0.2, columnspacing=0.8, bbox_to_anchor=(1.0, 1.12))
    fig.savefig('fig5_calib.png')

def fig6():
    """ablation: change relative to DRIFT"""
    A = S['abl']; names = [v for v in ['DRIFT-noE', 'DRIFT-noF', 'DRIFT-noR', 'DRIFT-noQ', 'DRIFT-noHmax', 'DRIFT-noShrink', 'DRIFT-noHead'] if v in A]
    fig, axs = plt.subplots(1, 3, figsize=(3.5, 1.75), sharey=True, gridspec_kw=dict(wspace=0.18, left=0.27, right=0.95, bottom=0.14, top=0.88))
    for ax, m, t in zip(axs, ['auc', 's20', 'tiw'], ['$\\Delta$AUROC', '$\\Delta$Sen@TiW$_{20}$', '$\\Delta$TiW (deployed)']):
        vals = [A[v][m][1] for v in names]
        ax.barh(range(len(names)), vals, color='0.55', edgecolor='black', lw=0.5, height=0.6)
        ax.axvline(0, color='black', lw=0.6); ax.set_title(t, fontsize=6.5)
        ax.tick_params(axis='x', labelsize=5.5); ax.locator_params(axis='x', nbins=3)
    axs[0].set_yticks(range(len(names))); NM = {'E': '$D^E$', 'F': '$D^F$', 'R': 'ref. risk', 'Q': 'quality', 'Hmax': 'hold bound', 'Shrink': 'shrinkage', 'Head': 'readout refit'}
    axs[0].set_yticklabels(['w/o ' + NM[n.replace('DRIFT-no', '')] for n in names], fontsize=6)
    axs[0].invert_yaxis()
    fig.savefig('fig6_ablation.png')

if __name__ == '__main__':
    for f in sys.argv[1:]: globals()[f](); print(f, 'ok')
