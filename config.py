"""Shared constants. Times are in seconds unless stated."""
WSEC = 4.0          # window length
SPH = 300.0         # seizure prediction horizon (intervention time)
SOP = 1800.0        # seizure occurrence period
LEAD_GAP = SPH + SOP  # a seizure is a lead event if the previous one ended >= this long before
INTER_GAP = 3600.0  # interictal training windows lie >= 1 h from any onset ...
POST_GAP = 1800.0   # ... and >= 30 min after any seizure end
SEQ = 32            # windows of causal context for the temporal network (128 s)
FP_WIN = 75         # firing-power window (5 min)
FP_FRAC = 0.5       # persistence: fraction of firing windows needed for an alarm
WARN = SPH + SOP    # warning duration = refractory period after an alarm
ALPHA = 0.20        # target interictal time-in-warning
DATA = './data/pat/'
RES = './results/'
