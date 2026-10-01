# Pretrained Audio Environment & Hardware Diagnostic Report
**AI Stethoscope Research Project**  
**Date:** September 12, 2026  
**Status:** Environment Inspection Complete  
**Scope:** Read-Only Hardware, Dependency, and Pretrained Model Feasibility Audit

---

## 1. System Hardware & Environment Specifications

| Component / Metric | Observed Specification | Details & Observations |
| :--- | :--- | :--- |
| **Operating System** | **Windows 11** (10.0.26200-SP0) | 64-bit AMD64 Architecture |
| **Python Version** | **Python 3.14.6** | Tag `v3.14.6:c63aec6`, MSC v.1944 64-bit |
| **PyTorch Version** | **2.14.0+cpu** | CPU-only PyTorch build |
| **CUDA Available (PyTorch)** | **False** | PyTorch does not have active CUDA bindings |
| **CUDA Version (PyTorch)** | **N/A** | No CUDA runtime bound to PyTorch |
| **Physical Hardware GPU(s)** | 1. **NVIDIA GeForce MX550** (2.0 GB VRAM)<br>2. **Intel Iris Xe Graphics** | The physical machine contains an entry-level NVIDIA GPU, but PyTorch is CPU-only, and 2 GB VRAM is severely memory-constrained for deep foundation models |
| **CPU Model** | **12th Gen Intel(R) Core(TM) i7-1255U** | 10 Cores (2 Performance + 8 Efficient), 12 Logical Processors |
| **Total System RAM** | **15.69 GB** (16,454,388 KB) | Standard DDR4/DDR5 system memory |
| **Free Physical RAM** | **~2.90 GB** (3,044,704 KB) | Current available headroom for model weights and forward buffers |
| **Project Drive (C:) Free Space** | **59.95 GB Free** (456.66 GB Total) | Ample storage for caching pretrained weights and feature matrices |

---

## 2. Python Package Installation Audit

| Package Name | Installation Status | Current Version | Notes & Compatibility |
| :--- | :---: | :---: | :--- |
| **`torch`** | **INSTALLED** | `2.14.0+cpu` | Operational CPU execution |
| **`torchaudio`** | ❌ **NOT INSTALLED** | — | Required for native torchaudio transforms |
| **`torchvision`** | ❌ **NOT INSTALLED** | — | Not present in environment |
| **`transformers`** | ❌ **NOT INSTALLED** | — | Required for Hugging Face AST/Wav2Vec2 pipelines |
| **`librosa`** | **INSTALLED** | `1.0.0` | Operational for DSP, STFT, Mel filtering |
| **`soundfile`** | **INSTALLED** | `0.14.0` | Operational for audio I/O |
| **`numpy`** | **INSTALLED** | `2.5.1` | Vectorized numerical computing |
| **`scipy`** | **INSTALLED** | `1.18.1` | Scientific computing & signal processing |
| **`scikit-learn`** | **INSTALLED** | `1.9.1` | Machine learning & evaluation metrics |
| **`pandas`** | **INSTALLED** | `3.0.3` | Dataframe manipulation & manifests |

---

## 3. Network & Hugging Face Reachability

* **Hugging Face Hub (`https://huggingface.co`):** **REACHABLE** (HTTP 200 response confirmed).
* **GitHub Raw Content (`https://raw.githubusercontent.com`):** **REACHABLE** (HTTP 200 response confirmed).
* **Assessment:** Automated downloads of official pretrained weights, model checkpoints, and configuration files will work reliably from this environment.

---

## 4. Existing Project Codebase Audit

An audit of `src/` and `scripts/` reveals:
1. **Existing Model Architectures:**
   * [`src/models/cnn_baseline.py`](file:///c:/Users/Tanay/Downloads/AI_Stethoscope/src/models/cnn_baseline.py) defines the custom modular 4-stage 2D CNN used in our earlier benchmark.
   * There is currently **no pretrained audio model code, pipeline, or wrapper** in `src/models/`.
2. **References in Research/Analysis Scripts:**
   * Design considerations in [`scripts/analyze_audio.py`](file:///c:/Users/Tanay/Downloads/AI_Stethoscope/scripts/analyze_audio.py), [`scripts/icbhi_padding_analysis.py`](file:///c:/Users/Tanay/Downloads/AI_Stethoscope/scripts/icbhi_padding_analysis.py), and [`analysis/full_cnn_baseline_results.md`](file:///c:/Users/Tanay/Downloads/AI_Stethoscope/analysis/full_cnn_baseline_results.md) previously outlined the theoretical suitability of AST, PANNs, and Wav2Vec2 at a standardized 16 kHz sampling rate.

---

## 5. Architectural Approach Recommendation

Given the hardware constraints (**CPU-only PyTorch execution**, **~2.9 GB free RAM**, and no dedicated multi-gigabyte GPU acceleration):

### Recommended Strategy: **A) Frozen Pretrained Audio Embeddings + Classifier / Multi-Task Head**

#### Why Approach A is Strongly Recommended:
1. **Computational Feasibility:** 
   * Extracting embeddings requires only a single **inference forward pass** ($O(N)$ with no gradients stored). On this 10-core Intel i7, forward extraction can be completed in ~10–15 minutes across all three datasets and cached.
   * In contrast, **Full Fine-Tuning (Approach B)** requires backpropagating through 80M+ parameters with optimizer momentum (AdamW requires ~3× parameter memory). On CPU, full fine-tuning would take **24–48+ hours** and would almost certainly trigger Out-Of-Memory (OOM) crashes on our ~2.9 GB available physical RAM.
2. **Memory Efficiency:**
   * Freezing the pretrained backbone allows batch sizes of 16–32 with negligible memory usage since no computation graph or intermediate layer activations need to be retained for backward passes.
3. **Multi-Task Synergies:**
   * Extracted frozen embeddings provide an immediate, rich representation that can feed directly into our multi-task architecture (shared feature pool projecting to Heart, ICBHI, and Kaggle heads).
4. **Reproducibility & Rigor:**
   * Pre-extracted embeddings provide deterministic representations for rapid hyperparameter tuning of multi-task classification heads without expensive backbone recomputations.

---

## 6. Recommended Pretrained Audio Model Candidates

### Candidate 1: PANNs (Pretrained Audio Neural Networks — CNN14 or CNN10)
* **Origin & Pretraining:** Kong et al. (Surrey University). Pretrained on Google AudioSet (~2 million clips, 527 sound classes).
* **Architecture:** 2D Convolutional Neural Network (CNN14: 14 layers, ~80M params; CNN10: 10 layers, ~5M params).
* **Input Compatibility:** Takes Log-Mel spectrograms $(128 \times T)$ directly at 16 kHz or 32 kHz, seamlessly matching our existing preprocessed spectrogram pipeline (`processed/full/`).
* **Hardware Profile on CPU:** Very fast forward inference on CPU, low memory footprint. PyTorch weights can be loaded directly via pure PyTorch (`torch.hub` or GitHub model definitions) without dependency on `torchaudio` or `transformers`.
* **Suitability for Auscultation:** Exceptional at detecting acoustic sound events (transient clicks, crackles, murmurs, wheezing harmonics) due to convolutional translation invariance across frequency bands.

### Candidate 2: AST (Audio Spectrogram Transformer)
* **Origin & Pretraining:** Yuan Gong et al. (MIT). Pure Vision Transformer (ViT) operating on non-overlapping 2D spectrogram patches $(16 \times 16)$, pretrained on ImageNet and AudioSet.
* **Architecture:** ViT-Base (~86M parameters).
* **Input Compatibility:** Expects Log-Mel spectrograms $(128 \times 1024)$ or interpolated patch grids. Fits naturally with our 128 Mel-frequency representation.
* **Hardware Profile on CPU:** Self-attention across time-frequency patches can run on CPU in frozen evaluation mode. Produces a 768-dimensional `[CLS]` token embedding.
* **Dependency Requirement:** Requires installing `transformers` (or implementing the AST model definition natively in PyTorch).
* **Suitability for Auscultation:** State-of-the-art benchmark standard across biomedical audio literature; captures long-range temporal dependencies (cardiac cycle periodicity, wheeze durations).

### Candidate 3: YAMNet (Yet Another Mobile Network)
* **Origin & Pretraining:** Google Research. MobileNet-v1 architecture trained on AudioSet-521.
* **Architecture:** Depthwise-separable 2D CNN (~3.7M parameters).
* **Input Compatibility:** Log-Mel spectrograms $(64 \times 96)$ at 16 kHz.
* **Hardware Profile on CPU:** Extremely lightweight (~3.7M params), ultra-fast CPU inference (<1 ms per window), minimal RAM consumption (<50 MB).
* **Suitability for Auscultation:** Ideal as an ultra-compact edge-deployable baseline to benchmark against PANNs and AST.

---

## 7. Next Steps Summary

1. Install the minimal necessary package if using AST (`transformers`) or use pure PyTorch model definitions for PANNs/CNN14 (which requires no extra packages).
2. Implement an embedding extractor module in `src/models/` or `src/features/` that loads pretrained weights, freezes the backbone, and generates fixed-length embeddings.
3. Cache the embeddings to `processed/pretrained_embeddings/` for Heart, ICBHI, and Kaggle.
4. Benchmark classification performance on the extracted embeddings.
