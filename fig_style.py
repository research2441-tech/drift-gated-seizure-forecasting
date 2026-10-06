import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.3, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'lines.linewidth': 1.0, 'savefig.dpi': 300,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
# grayscale identities: (face, edge, hatch, marker, linestyle)
SYS = {
 'SRC': ('white', 'black', '', 'o', (0, (1, 1))), 'HEAD': ('0.92', 'black', '', 's', (0, (2, 1))),
 'TENT': ('0.85', 'black', '....', 'v', (0, (3, 1, 1, 1))), 'STATIC': ('0.75', 'black', '', 'D', (0, (4, 2))),
 'ALIGN': ('0.6', 'black', '', '^', (0, (5, 1))), 'ALIGN-30': ('0.6', 'black', '', '<', (0, (2, 2))), 'SLOWALIGN': ('0.75', 'black', '', '<', (0, (2, 2))),
 'ALIGN+G': ('0.45', 'black', '///', '>', (0, (6, 2, 1, 2))), 'GAIN': ('0.3', 'black', '', 'P', (0, (8, 2))),
 'DRIFT': ('0.05', 'black', '', '*', '-'),
}
def panel(ax, s): ax.text(-0.16, 1.04, s, transform=ax.transAxes, fontsize=8, weight='bold', va='bottom')
