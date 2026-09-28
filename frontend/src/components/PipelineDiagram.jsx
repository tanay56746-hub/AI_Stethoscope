import React from 'react';

/**
 * PipelineDiagram Component
 * Visualizes the 7 sequential stages of the AI Stethoscope inference architecture.
 */
export default function PipelineDiagram() {
  const steps = [
    {
      num: 1,
      title: 'Audio Input',
      desc: 'WAV sound recording (stethoscope chest contact)',
    },
    {
      num: 2,
      title: 'Preprocessing',
      desc: '20 Hz HP filter, 16 kHz sinc resample, peak normalize, 5.0s fixed window',
    },
    {
      num: 3,
      title: 'Feature Extraction',
      desc: '128 × 313 Log-Mel spectrogram + 397 physical acoustic features',
    },
    {
      num: 4,
      title: 'Shared CNN',
      desc: 'Convolutional feature extractor learning local spectral auscultation patterns',
    },
    {
      num: 5,
      title: 'PANNs Backbone',
      desc: 'Frozen 2048-dim AudioSet CNN14 embedding projected to 128 dims',
    },
    {
      num: 6,
      title: 'Multi-Task Fusion',
      desc: 'Concatenation & shared latent routing across task branches',
    },
    {
      num: 7,
      title: 'Heart & Lung Results',
      desc: 'Independent Softmax outputs for Heart, Lung Acoustics, and Disease',
    },
  ];

  return (
    <div className="pipeline-section">
      <div className="pipeline-header">
        <h2>HOW THE AI WORKS</h2>
        <p>End-to-end multi-task deep learning auscultation pipeline from raw audio to clinical acoustics</p>
      </div>

      <div className="pipeline-steps-grid">
        {steps.map((s) => (
          <div key={s.num} className="pipeline-step-card">
            <div className="step-number">{s.num}</div>
            <div className="step-title">{s.title}</div>
            <div className="step-desc">{s.desc}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
