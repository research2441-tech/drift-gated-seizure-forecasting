import sys; sys.path.insert(0,'.')
import numpy as np, json, warnings; warnings.filterwarnings('ignore')
from semisynth import *
from train import prepare, fit_kF
pid='chb05'; others=prepare(['chb01','chb02','chb03','chb04','chb06'])
ref=SourceRef().fit(others); kF=fit_kF(others,ref,dict(MODES['DRIFT'],gate=('F',)))
P=labels(load(pid)); Q,onsets,kinds=inject(P,0.15,np.random.default_rng(0))
out={'t':Q['t'].tolist(),'onsets':onsets,'kinds':[(k,float(o)) for k,o in kinds]}
for name in ['ALIGN','GAIN','DRIFT','STATIC','SLOWALIGN']:
    c=dict(MODES[name]); 
    if c.get('gate'): c['k_F']=kF
    o=run(Q,None,ref,c,keep_inputs=True); s=score(o['inp'],o['usable'])
    out[name]={'score':np.nan_to_num(s).tolist(),'state':o['state'].tolist(),'gamma':o['gamma'][:,0].tolist(),'Q':o['Q'].tolist()}
json.dump(out,open('./results/fig2_trace.json','w'))
print(onsets,kinds)
