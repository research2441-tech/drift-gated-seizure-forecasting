import sys, exp2
from exp import subjects
G = exp2.G
exp2.DEPLOY = {'none': {'SRC': dict(head=False), 'HEAD': dict(head=True)},
               'zscore': {'ALIGN': dict(head=True)}, 'zslow': {'SLOWALIGN': dict(head=True)},
               'gain': {'GAIN': dict(head=True), 'DRIFT': dict(head=True, gate=G)}}
exp2.TRAIN = {k: v for k, v in exp2.TRAIN.items() if k in exp2.DEPLOY}
direction, seed = sys.argv[1], int(sys.argv[2])
chb = sorted(subjects('chb')); sie = sorted(subjects('siena'))
if direction == 'c2s':
    exp2.run_fold('x', 0, seed, train_groups=chb, target_groups=sie, tag=f'x_c2s_s{seed}')
else:
    exp2.run_fold('x', 1, seed, train_groups=sie, target_groups=chb, tag=f'x_s2c_s{seed}')
