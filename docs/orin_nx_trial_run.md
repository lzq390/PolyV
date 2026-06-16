# PolyV Orin NX 8G Trial Run Guide

This package is built on the local development machine, but dependencies are
installed on the Jetson board. Do not copy the local WSL, Windows, or conda
environment to the board.

## 1. Build the package locally

From the repository root:

```bash
git status --short
python -m unittest discover -s tests -p 'test*.py' -v
bash scripts/package_orin_trial.sh
```

The package is written to:

```text
dist/polyv-orin-trial-<commit>.tar.gz
dist/polyv-orin-trial-<commit>.sha256
```

The archive intentionally excludes `.git/`, `outputs/`, `dist/`, `.venv/`,
`__pycache__/`, and local video files.

## 2. Copy the package to the board

Example:

```bash
scp dist/polyv-orin-trial-<commit>.tar.gz <jetson-user>@<jetson-ip>:/data/polyv/
scp dist/polyv-orin-trial-<commit>.sha256 <jetson-user>@<jetson-ip>:/data/polyv/
```

For the current Orin NX board, the login target is `nvidia@192.168.31.245`.

On the board:

```bash
cd /data/polyv
sha256sum -c polyv-orin-trial-<commit>.sha256
tar -xzf polyv-orin-trial-<commit>.tar.gz
cd polyv-orin-trial-<commit>
```

## 3. Install board dependencies

Run this only on Jetson Linux:

```bash
./scripts/install_jetson.sh
```

The installer uses apt packages from the board OS:

- `python3-venv`
- `python3-opencv`
- `python3-numpy`
- `python3-pil`

It then creates `.venv` with `--system-site-packages` and runs the unit tests.

## 4. Run one video

Put input videos under `/data/polyv/videos`:

```bash
mkdir -p /data/polyv/videos /data/polyv/outputs
./scripts/run_video.sh /data/polyv/videos/sample.mp4
```

Expected outputs:

```text
/data/polyv/outputs/sample.result.json
/data/polyv/outputs/sample.time.log
```

The JSON result should contain:

- `state`
- `alert`
- `transition_time_sec`
- `confidence`
- `evidence`

## 5. Run with overlay images

```bash
./scripts/run_video.sh /data/polyv/videos/sample.mp4 --overlay
```

Expected extra output:

```text
/data/polyv/outputs/sample_overlay/
```

Use the overlay images to inspect ROI placement and the alert frame.

## 6. Record board performance

In another terminal on the board:

```bash
tegrastats --interval 1000
```

For a baseline run, record:

- command line used
- video duration and resolution
- result JSON
- `*.time.log`
- a short `tegrastats` excerpt
- current power mode from `sudo /usr/sbin/nvpmodel -q`

Use `sudo jetson_clocks` only for a separate maximum-performance benchmark.
The default trial run should reflect the board's normal power mode.

## Acceptance criteria

- Unit tests pass on the board.
- One representative video produces a complete JSON result.
- Overlay mode writes ROI/debug images.
- `state` and `alert` match the PC baseline for the same input video.
- `transition_time_sec` is within 1-2 seconds of the PC baseline.
