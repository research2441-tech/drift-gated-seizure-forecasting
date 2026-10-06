"""Systems compared in the study. All share architecture, training budget, alarm policy and the
online threshold recalibration; they differ in how the target stream is adapted."""
VARIANTS = {
    # cross-patient model, no adaptation of any kind
    'SRC':      dict(mode='none', gate=None, rel=False, head=False),
    # supervised readout adaptation from adjudicated outcomes only
    'HEAD':     dict(mode='none', gate=None, rel=False, head=True),
    # static patient calibration (first monitored hour, then frozen) + readout adaptation
    'STATIC':   dict(mode='zscore', gate=None, tau_a=1e12, freeze_after=900, head=True),
    # Tent-style test-time adaptation (entropy minimisation of encoder normalisation), ungated
    'TENT':     dict(mode='none', gate=None, rel=False, head=True, tent=True),
    # unrestricted continual alignment (per-feature running z-score and reference), never suspended
    'ALIGN':    dict(mode='zscore', gate=None, head=True),
    # restricted (gain-only) adaptation without the drift gate
    'GAIN':     dict(mode='gain', gate=None, head=True),
    # unrestricted alignment governed by the drift gate
    'ALIGN+G':  dict(mode='zscore', gate=('E', 'F', 'R'), head=True),
    # proposed: restricted adaptation + drift gate + bounded suspension + source shrinkage
    'DRIFT':    dict(mode='gain', gate=('E', 'F', 'R'), head=True),
}
