"""Assemble per-patient continuous window streams with seizure annotations.

Outputs data/pat/<pid>.npz with
  X      int16 [n,18,16]  raw quantised features (see io_feat)
  t      float64 [n]      window end time in seconds from the first recording start
  seg    int32 [n]        contiguous segment id (new id after a gap > 60 s)
  onset, offset float64   seizure times (s, same clock)
  avail  int8 [n,18]      channel availability from the montage
"""
import json, re, glob, os, datetime, numpy as np
from io_feat import read_feat_file, NC, NF, WSEC

UP = './features/'
import os as _os
OUT = _os.environ.get('PAT_OUT', './data/pat/')
os.makedirs(OUT, exist_ok=True)
META = json.load(open(UP + 'seizure_meta_txt.json'))

def chb_seizures():
    """{file_name: [(start_s, end_s), ...]} from the CHB-MIT summary files."""
    res = {}
    for k, txt in META.items():
        if 'summary' not in k: continue
        cur = None
        for line in txt.splitlines():
            m = re.match(r'File Name:\s*(\S+)', line)
            if m: cur = m.group(1); res[cur] = []; continue
            m = re.match(r'Seizure\s*\d*\s*Start Time:\s*(\d+)', line)
            if m and cur: res[cur].append([int(m.group(1)), None]); continue
            m = re.match(r'Seizure\s*\d*\s*End Time:\s*(\d+)', line)
            if m and cur: res[cur][-1][1] = int(m.group(1))
    return res

def hms(s):
    p = re.findall(r'\d+', s)[:3]
    p = [int(x) for x in p] + [0] * (3 - len(p))
    return p[0] * 3600 + p[1] * 60 + p[2]

SIENA_KEYS = ['PN00-1','PN00-2','PN00-3','PN00-4','PN00-5','PN01-1','PN03-1','PN03-2','PN05-2','PN05-3',
              'PN05-4','PN06-1','PN06-2','PN06-3','PN06-4','PN06-5','PN07-1','PN09-1','PN09-2','PN09-3',
              'PN10-1','PN10-10','PN10-2','PN10-3','PN10-4.5.6','PN10-7.8.9','PN11-1','PN12-1.2','PN12-3',
              'PN12-4','PN13-1','PN13-2','PN13-3','PN14-1','PN14-2','PN14-3','PN14-4','PN16-1','PN16-2',
              'PN17-1','PN17-2']

def siena_name(raw):
    n = raw.replace('PNO', 'PN0').replace('.edf', '')
    if n in SIENA_KEYS: return n + '.edf'
    if n.rstrip('-') + '-1' in SIENA_KEYS: return n.rstrip('-') + '-1.edf'
    raise KeyError(raw)

def siena_seizures():
    """{file_name: [(start_s_rel, end_s_rel)]} relative to the registration start (robust to the
    several layouts used in the Siena seizure lists)."""
    res = {}
    for k, txt in META.items():
        if 'Seizures-list' not in k: continue
        f = r0 = None; cur = None
        for line in txt.splitlines():
            m = re.search(r'File name:\s*(\S+)', line)
            if m: f = siena_name(m.group(1)); continue
            m = re.search(r'Registration start time:\s*([\d.:]+)', line)
            if m: r0 = hms(m.group(1)); continue
            if 'Registration' in line: continue
            m = re.search(r'start time:\s*([\d.:]+)', line, re.I)
            if m and f is not None and r0 is not None:
                cur = [(hms(m.group(1)) - r0) % 86400, None]; res.setdefault(f, []).append(cur); continue
            m = re.search(r'end time:\s*([\d.:]+)', line, re.I)
            if m and cur is not None and cur[1] is None:
                cur[1] = (hms(m.group(1)) - r0) % 86400
                if cur[1] < cur[0] or cur[1] - cur[0] > 900: cur[1] = cur[0] + 120   # malformed end times
    return {f: [(a, b if b is not None else a + 60) for a, b in v] for f, v in res.items()}

def build(files):
    chb = chb_seizures(); sie = siena_seizures()
    recs = {}
    for fp in files:
        for meta, a in read_feat_file(fp):
            name = meta['key'].split('/')[-1]
            pid = name.split('_')[0] if name.startswith('chb') else name.split('-')[0]
            pid = re.sub(r'[abc]$', '', pid)            # chb17a/b/c -> chb17
            recs.setdefault(pid, []).append((meta, a, name))
    summary = {}
    for pid, lst in recs.items():
        lst.sort(key=lambda r: (datetime.datetime.strptime(r[0]['start'], '%d.%m.%y %H.%M.%S'), r[0].get('rec0', 0)))
        t0 = datetime.datetime.strptime(lst[0][0]['start'], '%d.%m.%y %H.%M.%S')
        X, T, S, AV, on, off = [], [], [], [], [], []
        seg, last_end = -1, None
        seen_on = set(); tmax = -np.inf; last_name = None
        for meta, a, name in lst:
            st = (datetime.datetime.strptime(meta['start'], '%d.%m.%y %H.%M.%S') - t0).total_seconds() + meta.get('rec0', 0) * meta['dur']
            fst = st - meta.get('rec0', 0) * meta['dur']
            # a different recording whose header clock overlaps the previous one (anonymised dates):
            # place it after the previous material with a 1-h gap, shifting its seizures consistently
            if name != last_name and st < tmax - 1:
                shift = (tmax - st) + 3600.0; st += shift; fst += shift
            last_name = name
            n = a.shape[0]
            if last_end is None or st - last_end > 60: seg += 1
            tt = st + WSEC * (np.arange(n) + 1)
            keep = tt > tmax + 0.5                                              # drop windows already covered
            if keep.any(): tmax = max(tmax, tt[keep].max())
            X.append(a[keep]); T.append(tt[keep]); S.append(np.full(int(keep.sum()), seg, np.int32))
            AV.append(np.tile(np.array(meta['avail'], np.int8), (int(keep.sum()), 1)))
            last_end = max(last_end or -np.inf, st + meta['nrec'] * meta['dur'])
            for (s0, s1) in (chb.get(name, []) if pid.startswith('chb') else sie.get(name, [])):
                if fst + s0 >= st - 1e-6 and fst + s0 <= st + meta['nrec'] * meta['dur'] and round(fst + s0) not in seen_on:
                    seen_on.add(round(fst + s0)); on.append(fst + s0); off.append(fst + (s1 if s1 else s0 + 30))
        X = np.concatenate(X); T = np.concatenate(T); S = np.concatenate(S); AV = np.concatenate(AV)
        o = np.argsort(on)
        np.savez(OUT + pid + '.npz', X=X, t=T, seg=S, avail=AV,
                 onset=np.array(on)[o], offset=np.array(off)[o])
        summary[pid] = dict(hours=round(len(T) * WSEC / 3600, 2), n_seizures=len(on), n_files=len(lst),
                            n_seg=int(seg + 1), avail_ch=int(AV.min(0).sum()))
    return summary

if __name__ == '__main__':
    import sys
    s = build(sorted(glob.glob(UP + 'feat*.bin')))
    old = json.load(open(OUT + 'summary.json')) if os.path.exists(OUT + 'summary.json') else {}
    old.update(s); json.dump(old, open(OUT + 'summary.json', 'w'), indent=1)
    for k, v in sorted(s.items()): print(k, v)
