# AI Stethoscope — Real-Time Training Monitor Dashboard

A lightweight, zero-dependency, read-only web dashboard for continuously monitoring the training progress and benchmark evaluation of the **AI Stethoscope** Stage 5B custom CNN baselines.

---

## Features

- **100% Read-Only**: Completely non-invasive. Does not touch, pause, restart, lock, or interfere with active training processes.
- **Pure Python Standard Library**: Requires **zero third-party dependencies** (`pip install` is not needed). Runs directly on any Python 3 installation using built-in `http.server`.
- **Live Auto-Refresh**: Web UI polls the local JSON status endpoint every **3 seconds** and updates UI metrics, progress bars, and badges smoothly.
- **Heart CNN Live Tracking**:
  - Process state: `Running`, `Completed`, or `Stopped / Idle`
  - Active Epoch / Total Epochs (e.g., `Epoch 9 / 15`)
  - Sub-epoch Batch Progress (e.g., `Step 400 / 598` with percentage bar)
  - Overall Model Progress percentage
  - Running Training Loss and Validation Loss
  - Validation Macro F1, Balanced Accuracy, and Patient-level F1
  - Best model epoch and best validation score
  - Verified presence of `best_model.pt` and `last_checkpoint.pt`
  - Real elapsed runtime and calculated ETA (based on actual observed epoch durations)
- **ICBHI CNN Baseline Overview**:
  - Training status (Epoch 3 checkpoint saved)
  - Validation & checkpoint status (`best_model.pt` from Epoch 2)
  - Held-out test evaluation status (`manifests/icbhi_test.csv`)
  - Test Window Macro F1, Patient Macro F1, and loss
- **Kaggle CNN Baseline Overview**:
  - Training status (9 epochs, early stopped)
  - Test evaluation status
  - Window Macro F1 and Patient Macro F1 (soft voting aggregation)
- **Overall Stage 5B Milestone Tracker**:
  - Real-time progress bar across all 6 benchmark pipeline milestones.

---

## How to Launch the Dashboard

Open a terminal or PowerShell prompt in the project root directory and run:

```bash
python monitor/dashboard.py
```

### Optional Arguments

You can specify a custom port or host if desired:

```bash
python monitor/dashboard.py --port 8080 --host 127.0.0.1
```

If port `8080` is already in use, the dashboard automatically binds to the next available port (`8081`, `8082`, etc.).

---

## Opening the Dashboard

Once started, open your web browser and navigate to:

```text
http://127.0.0.1:8080
```
*(or `http://localhost:8080`)*

---

## How to Stop the Dashboard

Press `Ctrl + C` in the terminal running `dashboard.py`. Stopping the dashboard will **not** affect the active training job.
