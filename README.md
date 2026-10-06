# Gain-Restricted, Drift-Gated Adaptation for Cross-Patient Seizure Forecasting

Code accompanying the manuscript *"Preserving Preictal Change and Bounding False-Warning Burden in Cross-Patient Seizure Forecasting with Gain-Restricted, Drift-Gated Adaptation"*.

The forecaster (DRIFT) handles adaptation in four parts:

- **Gain restriction.** Fast adaptation is confined to a per-channel gain direction.
- **Gated shape reference.** Spectral shape is referred to a slow patient reference, which is updated only when a drift gate allows it. The gate combines signal quality, source-expert disagreement and shape change.
- **Unavailable state.** Corrupted windows return an explicit *unavailable* state.
- **Policy-level threshold calibration.** The alarm threshold of the complete policy (firing power, persistence, refractory period) is calibrated with Learn-then-Test. Calibration is first done on held-out source patients and then on adjudicated seizure-free hours.

All files live in one flat folder.

## Requirements

- Python 3.11 or later; package versions are in `requirements.txt`.
- Runs on CPU only; no GPU is needed.
- Feature extraction uses a modern browser (Chrome or Edge) and `edf_feature_extractor.js`.

```
pip install -r requirements.txt
```

## Data

The data are public EDF recordings from PhysioNet:

- **CHB-MIT Scalp EEG Database v1.0.0:** `chbmit/1.0.0/...`
- **Siena Scalp EEG Database v1.0.0:** `siena-scalp-eeg/1.0.0/...`

The study used the following portion of each database:

- **CHB-MIT, complete:** recordings of chb01–chb07.
- **CHB-MIT, excerpts:** for chb08–chb24, seizure-centred excerpts from 105 min before onset to 30 min after it, plus one seizure-free hour.
- **Siena:** the first two seizure intervals of every patient.

## Pipeline

1. **Feature extraction (browser).**
   - Open any page on `https://physionet-open.s3.amazonaws.com/` and paste `edf_feature_extractor.js` into the developer console.
   - Then call `await runTasks2('feat_part1', [[key, start_s, end_s], ...], 3)`.
     - `key` is the EDF path, for example `chbmit/1.0.0/chb01/chb01_03.edf`.
     - `start_s` and `end_s` are seconds within the file. Omit them to process the whole file.
   - The browser saves the result as `feat_part1.bin`.
   - Each 4-s window gives 18 bipolar derivations × 16 int16 features:
     - 13 log band powers covering 0.5–47 Hz and 65–95 Hz;
     - log line length;
     - log peak amplitude;
     - flat fraction.
   - Siena referential channels are re-referenced to the same bipolar montage.

2. **Patient streams.**
   - Put the `.bin` files in `./features/` together with `seizure_meta_txt.json`, which holds the CHB-MIT summary files and the Siena seizure lists keyed by file name.
   - Run `python build_dataset.py`. It writes `./data/pat/<patient>.npz`.

3. **Patient-held-out experiments (CHB-MIT, four folds, two seeds).**
   ```
   NT=2 python exp2.py chb 0,1,2,3 0,1
   python make_results.py            # -> results/summary_chb.json
   ```

4. **Cross-dataset transfer.**
   ```
   python run_transfer.py c2s 0
   python run_transfer.py s2c 0
   ```

5. **Additional analyses.**
   - **Semi-synthetic preservation test:** `python run_semisynth.py chb01,chb02,chb03,chb04,chb05,chb06,chb07`
   - **Disturbance injection:** `python run_disturb.py`
   - **Alarm-policy sensitivity:** `python sens_policy.py`
   - **Run time:** `python effic.py`
   - **Leakage check (time since segment start):** `python confound.py`

6. **Figures.**
   - `diagram.py` (Fig. 1).
   - `fig2_data.py` followed by `fig2.py`.
   - `fig_semisynth.py`.
   - `python fig_results.py fig4 fig5 fig6`.
   - `fig7.py`, `fig8.py`.

## File map

| File | Content |
|---|---|
| `config.py` | Shared constants: horizon, occurrence period, paths |
| `io_feat.py`, `build_dataset.py` | Feature-file reader; patient stream assembly with seizure annotations |
| `prep.py` | Loading, labels, channel quality (flat, amplitude, gain-invariant muscle ratio) |
| `source_ref.py` | Source statistics and per-channel gain estimate along the gain direction |
| `model.py` | Montage-aware encoder (masked channel attention) and causal TCN expert |
| `train.py` | Patient-bagged source training of K experts |
| `deploy.py` | Online engine: restricted adaptation, drift gate, unavailable state, readout refit, threshold recalibration |
| `alarm.py` | Firing power, alarm policy, metrics, Hoeffding–Bentkus p-values, Learn-then-Test |
| `evaluate.py`, `exp.py`, `exp2.py`, `variants.py` | Fold construction, baselines (SRC, HEAD, TENT, STATIC, ALIGN, SLOWALIGN, ALIGN+G, GAIN) and ablations |
| `analyze.py`, `make_results.py` | Wilcoxon signed-rank tests, Holm correction, rank-biserial effect size, patient bootstrap |
| `semisynth.py`, `disturb.py`, `run_*.py`, `sens_policy.py`, `effic.py`, `confound.py` | Additional experiments |
| `fig_*.py`, `fig2.py`, `fig7.py`, `fig8.py`, `diagram.py` | Grayscale figures |

## Notes

- All reported numbers come from the scripts above; no result was tuned on the test patients.
- The risk-control guarantee assumes exchangeable hour-blocks. Serially dependent EEG does not satisfy this assumption, so the guarantee is approximate.
- The code is for research only and is not a medical device.

## License

MIT for the code. The data remain under their PhysioNet licences (Open Data Commons Attribution).
