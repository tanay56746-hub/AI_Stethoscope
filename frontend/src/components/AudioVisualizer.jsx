// =============================================================================
// [LAYER: FRONTEND COMPONENT] | File: frontend/src/components/AudioVisualizer.jsx
// Purpose: HTML5 Canvas Time-Domain Waveform & Log-Mel Spectrogram Renderer
// =============================================================================
import React, { useEffect, useRef } from 'react';

/**
 * AudioVisualizer Component
 * Renders the preprocessed audio waveform and Log-Mel spectrogram on HTML5 Canvases.
 */
export default function AudioVisualizer({ waveform, spectrogram, duration = 5.0 }) {
  const waveformCanvasRef = useRef(null);
  const specCanvasRef = useRef(null);

  // Render Waveform
  useEffect(() => {
    const canvas = waveformCanvasRef.current;
    if (!canvas || !waveform || waveform.length === 0) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    // Draw clinical grid lines
    ctx.strokeStyle = '#f1f5f9';
    ctx.lineWidth = 1;
    // horizontal grid lines
    for (let y = 20; y < height; y += 25) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    // vertical grid lines (every 1 second)
    const secondStep = width / 5;
    for (let s = 1; s <= 4; s++) {
      const x = s * secondStep;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    // Center baseline
    const centerY = height / 2;
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(width, centerY);
    ctx.stroke();

    // Plot audio waveform
    ctx.strokeStyle = '#0f766e';
    ctx.lineWidth = 1.5;
    ctx.beginPath();

    const step = width / (waveform.length - 1);
    waveform.forEach((val, i) => {
      const x = i * step;
      // Amplitude scaled to canvas height with padding
      const y = centerY - val * (centerY * 0.85);
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

  }, [waveform]);

  // Render Log-Mel Spectrogram
  useEffect(() => {
    const canvas = specCanvasRef.current;
    if (!canvas || !spectrogram || !spectrogram.matrix || spectrogram.matrix.length === 0) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    const matrix = spectrogram.matrix; // [n_freq_bins, n_time_frames]
    const nFreq = matrix.length;
    const nTime = matrix[0].length;

    // Create offscreen image buffer for efficient pixel rendering
    const imgData = ctx.createImageData(width, height);
    const data = imgData.data;

    // Biomedical spectrogram colormap: Deep Navy -> Teal -> Amber/Yellow -> Off-white
    const getColormapColor = (val) => {
      // val in [0.0, 1.0]
      const clamped = Math.max(0, Math.min(1, val));
      let r, g, b;

      if (clamped < 0.25) {
        // Deep Navy (#0f172a to #0f766e)
        const t = clamped / 0.25;
        r = Math.round(15 + t * (15 - 15));
        g = Math.round(23 + t * (118 - 23));
        b = Math.round(42 + t * (110 - 42));
      } else if (clamped < 0.65) {
        // Teal to Medical Amber (#0f766e to #d97706)
        const t = (clamped - 0.25) / 0.4;
        r = Math.round(15 + t * (217 - 15));
        g = Math.round(118 + t * (119 - 118));
        b = Math.round(110 + t * (6 - 110));
      } else {
        // Amber to Bright Yellow/White (#d97706 to #fef08a)
        const t = (clamped - 0.65) / 0.35;
        r = Math.round(217 + t * (254 - 217));
        g = Math.round(119 + t * (240 - 119));
        b = Math.round(6 + t * (138 - 6));
      }

      return [r, g, b];
    };

    // Fill each canvas pixel
    for (let py = 0; py < height; py++) {
      // Invert Y so low frequencies (index 0) are at the bottom of the canvas
      const freqIdx = Math.floor((1 - py / height) * nFreq);
      const safeFreq = Math.min(nFreq - 1, Math.max(0, freqIdx));

      for (let px = 0; px < width; px++) {
        const timeIdx = Math.floor((px / width) * nTime);
        const safeTime = Math.min(nTime - 1, Math.max(0, timeIdx));

        const val = matrix[safeFreq][safeTime];
        const [r, g, b] = getColormapColor(val);

        const pixelIndex = (py * width + px) * 4;
        data[pixelIndex] = r;
        data[pixelIndex + 1] = g;
        data[pixelIndex + 2] = b;
        data[pixelIndex + 3] = 255; // Alpha
      }
    }

    ctx.putImageData(imgData, 0, 0);

  }, [spectrogram]);

  return (
    <div>
      {/* Waveform Card */}
      <div className="viz-card">
        <div className="viz-title-row">
          <div className="viz-title">AUDIO WAVEFORM</div>
          <div className="viz-caption">16 kHz • 5.0s Peak-Normalized Auscultation</div>
        </div>
        <div className="canvas-container">
          <canvas
            ref={waveformCanvasRef}
            className="viz-canvas"
            width={800}
            height={130}
          />
        </div>
        <div className="viz-legend-bar">
          <span>0.0s</span>
          <span>1.0s</span>
          <span>2.0s</span>
          <span>3.0s</span>
          <span>4.0s</span>
          <span>5.0s</span>
        </div>
      </div>

      {/* Log-Mel Spectrogram Card */}
      <div className="viz-card">
        <div className="viz-title-row">
          <div className="viz-title">LOG-MEL SPECTROGRAM</div>
          <div className="viz-caption">128 Mel Bins • 20 Hz – 2000 Hz Clinical Band</div>
        </div>
        <div className="canvas-container">
          <canvas
            ref={specCanvasRef}
            className="viz-canvas"
            width={800}
            height={150}
          />
        </div>
        <div className="viz-legend-bar">
          <span>Low Freq (20 Hz)</span>
          <span>Visualization of preprocessed audio representation</span>
          <span>High Freq (2000 Hz)</span>
        </div>
      </div>
    </div>
  );
}
