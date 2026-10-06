"""Reader for the compact EEG window-feature files produced by extract.js.

Each file is a concatenation of records:
  uint32 json_len, uint32 n_win, json header, zero pad to 4 bytes,
  int16[n_win, 18, 16] window features.
Features per bipolar channel (4 s windows, no overlap):
  0-12  log10 PSD (uV^2/Hz) in bands 0.5-2,2-4,4-6,6-8,8-10,10-13,13-16,16-20,
        20-25,25-30,30-38,38-47,65-95 Hz (x1000)
  13    log10 mean |diff| (line length per sample, x1000)
  14    log10 max |x| (x1000)
  15    fraction of zero first differences (x10000)
Unavailable channels are filled with -32768.
"""
import json, numpy as np
from urllib.parse import unquote

BIP = ['FP1-F7','F7-T7','T7-P7','P7-O1','FP1-F3','F3-C3','C3-P3','P3-O1','FP2-F4','F4-C4',
       'C4-P4','P4-O2','FP2-F8','F8-T8','T8-P8','P8-O2','FZ-CZ','CZ-PZ']
BANDS = [(0.5,2),(2,4),(4,6),(6,8),(8,10),(10,13),(13,16),(16,20),(20,25),(25,30),(30,38),(38,47),(65,95)]
NC, NF, WSEC = 18, 16, 4.0

def read_feat_file(path):
    buf = open(path, 'rb').read()
    off, out = 0, []
    while off < len(buf):
        jl, nw = np.frombuffer(buf, np.uint32, 2, off); off += 8
        meta = json.loads(buf[off:off+jl].decode()); off += jl + ((4 - jl % 4) % 4)
        n = int(nw) * NC * NF
        a = np.frombuffer(buf, np.int16, n, off).reshape(int(nw), NC, NF); off += 2 * n
        meta['key'] = unquote(meta['key'])
        out.append((meta, a))
    return out

def decode(a):
    """int16 -> float32 features, NaN where channel unavailable."""
    x = a.astype(np.float32)
    miss = (a[..., 0] == -32768)
    x[..., :15] /= 1000.0
    x[..., 15] /= 10000.0
    x[miss] = np.nan
    return x
