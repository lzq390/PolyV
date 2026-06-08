# PolyV Gel Climbing Detector

Two-stage fixed-camera detector for bottle stirring videos:

- `LIQUID_STIRRING`
- `FINAL_GEL_ROD_CLIMBING`

The detector intentionally treats early foam and local whitening as
`LIQUID_STIRRING`. It alerts when gel wrapping around the stirring rod becomes
persistent across a rolling window. Exact 8min vs 9min separation is not the
goal; the goal is to separate the clearly gelled/rod-climbing phase from the
earlier liquid stirring phase.

## Install

```bash
conda env create -f environment.yml
conda activate CV
```

If `CV` already exists:

```bash
conda env update -n CV -f environment.yml
```

## Run

```bash
conda run -n CV python -m polyv_detector.cli \
  --video "/mnt/c/Users/ASUS/Downloads/VID_20260603_151103 - Trim.mp4" \
  --sample-fps 1 \
  --json-out result.json
```

Example output from the current sample video:

```json
{
  "state": "FINAL_GEL_ROD_CLIMBING",
  "alert": true,
  "transition_time_sec": 568.976,
  "confidence": 0.75,
  "evidence": {
    "white_coverage": 0.5152,
    "sparse_hole_ratio": 0.4958,
    "rod_wrap_height_px": 326,
    "rod_connection_score": true,
    "stable_duration_sec": 59.997,
    "stable_candidate_ratio": 0.8033
  }
}
```

## Defaults

The default ROIs are calibrated from the current 1920x1080 sample video and are
stored as normalized coordinates, so resized copies of the same camera view keep
the same geometry.

Final alert conditions:

- `white_coverage >= 0.45`
- `rod_wrap_height_px >= 300`
- final-candidate frames fill at least 80% of a 60-second window
- sparse-hole ratio is emitted as evidence, not used as a hard gate, because
  glass/background reflections make it unstable in this camera view

After alerting, the state locks to `FINAL_GEL_ROD_CLIMBING` to avoid repeat
alerts from stirring fluctuations.
