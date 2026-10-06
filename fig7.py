import json, numpy as np
from fig_style import *
SP = json.load(open('./results/sens_policy.json'))
fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.95), gridspec_kw=dict(wspace=0.45, left=0.13, right=0.98, bottom=0.27, top=0.9))
ax = axs[0]
for i, a in enumerate([0.1, 0.2, 0.3]):
    v = SP[f'alpha={a}']; long = [p for p in v if v[p]['inter_h'] >= 24]
    y = [v[p]['tiw'] for p in long]
    x = a + np.random.default_rng(i).uniform(-0.012, 0.012, len(y))
    ax.scatter(x, y, s=10, facecolor='0.6', edgecolor='black', lw=0.4, zorder=3)
    ax.plot([a - 0.025, a + 0.025], [np.median(y)] * 2, color='black', lw=1.2)
ax.plot([0.05, 0.35], [0.05, 0.35], ls='--', color='0.4', lw=0.7)
ax.text(0.33, 0.30, 'TiW $=\\alpha$', fontsize=6, ha='right', va='top', color='0.25')
ax.set_xlim(0.05, 0.35); ax.set_ylim(-0.01, 0.36); ax.set_xticks([0.1, 0.2, 0.3])
ax.set_xlabel('Target $\\alpha$'); ax.set_ylabel('Realised interictal TiW')
ax = axs[1]
labels = [('frac=0.3', 'FP 0.3'), ('alpha=0.2', 'FP 0.5'), ('frac=0.7', 'FP 0.7'), ('win=45', '3 min'), ('win=150', '10 min')]
vals = []
for k, _ in labels:
    v = SP[k]; vals.append(np.nanmean([v[p]['sens'] for p in v]))
ax.bar(range(5), vals, color=['0.85', '0.35', '0.85', '0.6', '0.6'], edgecolor='black', lw=0.5, width=0.65)
ax.set_xticks(range(5)); ax.set_xticklabels([l for _, l in labels], rotation=35, ha='right', fontsize=6)
ax.set_ylabel('Deployed sensitivity'); ax.set_ylim(0, 0.25)
for i, v in enumerate(vals): ax.text(i, v + 0.006, f'{v:.2f}', ha='center', fontsize=5.6)
panel(axs[0], '(a)'); panel(axs[1], '(b)')
fig.savefig('fig7_policy.png'); print('ok')
