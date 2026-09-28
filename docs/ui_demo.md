# AI Stethoscope Demonstration Interface Guide

This document describes the architecture, startup procedures, and demonstration workflow for the **AI Stethoscope Demonstration UI** (Stage 3).

---

## 1. Overview & Architecture

The demonstration interface connects a modern React frontend to the verified offline multi-task inference pipeline via a lightweight Python API server.

```
Frontend (React + Vite, Port 5173 / 8000)
    ↓  POST /api/predict (Multipart WAV or sample path)
Backend API Server (scripts/demo_server.py, Port 8000)
    ↓  Cached Singleton Engine (scripts/inference_engine.py)
Offline Inference Pipeline (scripts/predict_audio.py)
    ↓  Feature Extraction (16 kHz Resampling, HP Filter, Peak Norm, 5s Window)
Log-Mel (128×313) + Physical Features (397-D) + PANNs CNN14 (2048-D)
    ↓
Verified Multi-Task Model (experiments/multitask_ablations/cnn_physical_panns/checkpoints/best_checkpoint.pt)
    ↓
Three Output Heads:
  1. Heart Sound Classification (Normal / Abnormal)
  2. Lung Acoustic Classification (Normal / Crackle / Wheeze / Crackle+Wheeze)
  3. Lung Disease Classification (Normal / Asthma / Other)
    ↓
JSON Response (Predictions, Probabilities, Downsampled Waveform & Spectrogram)
    ↓
Interactive Biomedical UI Dashboard (Waveform Canvas, Spectrogram Canvas, Probability Bars)
```

---

## 2. Project Structure

```
AI_Stethoscope/
├── frontend/                     # React + Vite Biomedical Demonstration UI
│   ├── dist/                     # Production static build
│   ├── src/
│   │   ├── components/
│   │   │   ├── AudioUploadCard.jsx   # Drag-and-drop WAV upload & sample selector
│   │   │   ├── AudioVisualizer.jsx   # Waveform & Log-Mel spectrogram canvases
│   │   │   ├── ModelInformation.jsx # Collapsible technical specs table
│   │   │   ├── PipelineDiagram.jsx  # "How the AI Works" 7-stage diagram
│   │   │   └── ResultCards.jsx      # Heart, Lung Acoustic, and Disease cards
│   │   ├── App.jsx               # Main dashboard layout and state management
│   │   ├── index.css             # Light clinical biomedical design system
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js            # Dev proxy configured to localhost:8000
│
├── scripts/
│   ├── demo_server.py            # Zero-dependency Python HTTP API server
│   ├── inference_engine.py       # Cached inference engine with visualization data
│   ├── predict_audio.py          # Baseline offline inference script
│   ├── preprocess_audio.py       # AudioPreprocessor (16 kHz, filter, 5s, log-mel)
│   └── test_panns_cnn14.py       # PANNs CNN14 architecture definition
│
├── pretrained_weights/
│   └── Cnn14_mAP=0.431.pth       # Official PANNs CNN14 weights
│
├── experiments/multitask_ablations/cnn_physical_panns/
│   └── checkpoints/
│       └── best_checkpoint.pt    # Verified best multi-task checkpoint
│
└── docs/
    └── ui_demo.md                # This demonstration guide
```

---

## 3. How to Start the Application

### Option A: Unified Server (Recommended for Professor Demonstration)

The Python backend automatically serves the built frontend (`frontend/dist`) directly from port 8000. Only one command is needed:

```bash
python scripts/demo_server.py --port 8000
```

Then open your browser at:
```
http://127.0.0.1:8000
```

### Option B: Development Mode (Hot Reloading)

If modifying frontend React components with instant hot reload:

1. **Start the Backend API Server:**
   ```bash
   python scripts/demo_server.py --port 8000
   ```

2. **In a separate terminal, start the Vite Dev Server:**
   ```bash
   cd frontend
   npm run dev
   ```

3. **Open Vite URL:**
   ```
   http://127.0.0.1:5173
   ```
   *(Requests to `/api/*` are automatically forwarded to port 8000 via Vite's proxy).*

---

## 4. How the UI Calls the ML Pipeline

1. **Preloading / Caching**: On backend launch, `StethoscopeInferenceEngine` initializes once and loads both `Cnn14_mAP=0.431.pth` (PANNs backbone) and `best_checkpoint.pt` (Multi-Task Stethoscope model) into memory. Subsequent inference calls take only ~0.5 to 0.7 seconds on CPU.
2. **Audio Input**: The user can drag-and-drop any WAV audio recording, browse local files, or click the preloaded sample (`b0032.wav`).
3. **Execution**: Clicking **ANALYZE AUDIO** sends a `POST /api/predict` request.
4. **Processing**:
   - `AudioPreprocessor` applies DC removal, 20 Hz 4th-order Butterworth high-pass filter, sinc resampling to 16 kHz, 0.95 peak normalization, and fixed 5.0-second windowing.
   - Computes Log-Mel spectrogram (128 bins × 313 frames) and extracts 397 physical acoustic features.
   - Preprocesses audio at 32 kHz and extracts 2048-dimensional frozen PANNs embedding.
   - Forward pass through `FinalMultiTaskStethoscopeModel` with PyTorch evaluation mode (`torch.no_grad()`).
5. **Visualization Generation**:
   - Generates 300 downsampled peak-envelope waveform points for the waveform canvas.
   - Generates downsampled normalized 64×105 spectrogram matrix for the Log-Mel heatmap canvas.
6. **Rendering**:
   - Probabilities for Heart (Normal vs Abnormal), Lung Acoustic (Normal, Crackle, Wheeze, Crackle+Wheeze), and Disease (Normal, Asthma, Other) are rendered as real-time probability outputs generated by the trained multitask model, with highest-probability classes highlighted.
   - The UI clearly distinguishes **Live Demonstration** (predictions generated for the uploaded audio as demonstration outputs, not evidence of model accuracy or clinical diagnosis) from **Model Benchmark Results** (formal performance established via held-out test-set evaluation).
   - Medical disclaimer is prominently displayed ("Research prototype — not a clinical diagnosis").

---

## 5. Demonstration Walkthrough Steps

To present the project to a faculty panel or professor:

1. **Open the Application**: Navigate to `http://127.0.0.1:8000`.
2. **Highlight the System Status**:
   - Point out the `● MODEL READY` badge.
   - Review the status bar: `Model: Multitask CNN`, `Audio: 16 kHz`, `Window: 5 seconds`, `Input: WAV`.
3. **Demonstrate Inference**:
   - Click the test sample `b0032.wav (Heart Sound - PhysioNet Challenge)` or drag an external WAV file.
   - Click **ANALYZE AUDIO**.
   - Note the real-time execution indicator (`ANALYZING AUDIO...`) and completion in ~0.6s.
4. **Examine Three-Task Output Cards**:
   - **Heart Sound Analysis**: Abnormal (59.33%), Normal (40.67%).
   - **Lung Acoustic Analysis**: Wheeze (41.81%), Normal (30.23%), Crackle+Wheeze (14.14%), Crackle (13.81%).
   - **Lung Disease Classification**: Asthma (40.44%), Other (34.30%), Normal (25.26%).
   - Real-time probability outputs generated by the trained multitask model for the demonstration audio.
   - Emphasize to the professor the distinction between the **Live Demonstration** outputs and the offline **Model Benchmark Results** on held-out test sets.

5. **Review Audio Visualizations**:
   - Observe the **Audio Waveform** with clinical grid and 5.0s time axis.
   - Observe the **Log-Mel Spectrogram** rendering the 20 Hz – 2000 Hz auscultation frequency distribution.
6. **Walkthrough "How the AI Works"**:
   - Scroll to the 7-step architecture diagram explaining the transformation from raw acoustic pressure to multi-task clinical classifications.
7. **Expand "Model Information & Technical Specifications"**:
   - Click to expand the verified specifications table detailing model dimensions, 397 acoustic features, 2048-dim PANNs embeddings, AdamW optimizer, and checkpoint path.
