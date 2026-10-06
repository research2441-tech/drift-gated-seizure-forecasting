"""E-sensitivity: alarm-policy constants (alpha, persistence, firing-power window) on stored
DRIFT-noHead streams, whose risk does not depend on adjudicated labels, so online
threshold recalibration can be replayed exactly offline."""
import json, glob, os, numpy as np, warnings
warnings.filterwarnings('ignore')
import alarm, config
from exp import get
from evaluate import online_thetas
from alarm import metrics, policy_online

def run_setting(alpha, frac, win):
    alarm.FP_FRAC = frac; alarm.FP_WIN = win
    out = {}
    for f in sorted(glob.glob(config.RES + 'runs/chb_f*_s0/DRIFT-noHead/*.npz')):
        pid = os.path.basename(f)[:-4]; P = get(pid); z = np.load(f)
        meta = json.load(open(os.path.dirname(f) + '/meta.json'))
        risk = z['risk'].astype(float)
        th, _ = online_thetas(P, risk, meta['theta_S'], alpha=alpha)
        m = metrics(policy_online(risk, P['t'], P['seg'], th), P['t'], P['onset'], P['offset'], P['lead'], risk)
        out[pid] = dict(sens=m['sens'], fa=m['fa_h'], tiw=m['tiw'], inter_h=m['inter_h'])
    alarm.FP_FRAC = 0.5; alarm.FP_WIN = 75
    return out

if __name__ == '__main__':
    R = {}
    for alpha in (0.1, 0.2, 0.3):
        R[f'alpha={alpha}'] = run_setting(alpha, 0.5, 75)
    for frac in (0.3, 0.7):
        R[f'frac={frac}'] = run_setting(0.2, frac, 75)
    for win in (45, 150):
        R[f'win={win}'] = run_setting(0.2, 0.5, win)
    json.dump(R, open(config.RES + 'sens_policy.json', 'w'))
    for k, v in R.items():
        long = [p for p in v if v[p]['inter_h'] >= 24]
        print(k, 'TiW med %.3f' % np.median([v[p]['tiw'] for p in long]), 'ok %.2f' % np.mean([v[p]['tiw'] <= float(k.split('=')[1]) + 1e-9 if k.startswith('alpha') else v[p]['tiw'] <= 0.2 + 1e-9 for p in long]),
              'sens %.3f' % np.nanmean([v[p]['sens'] for p in v]))
