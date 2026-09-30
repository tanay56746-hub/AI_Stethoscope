// =============================================================================
// [LAYER: VERCEL SERVERLESS BRIDGE] | File: api/status.js
// Purpose: Cloud Serverless Model Status & Specs Endpoint (/api/status)
// =============================================================================

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  if (req.method === 'OPTIONS') return res.status(200).end();

  const backendUrl = process.env.EXTERNAL_BACKEND_URL || process.env.RENDER_BACKEND_URL;
  if (backendUrl) {
    try {
      const targetUrl = `${backendUrl.replace(/\/$/, '')}/api/status`;
      const response = await fetch(targetUrl, {
        headers: { 'Accept': 'application/json' },
      });
      if (response.ok) {
        const data = await response.json();
        return res.status(200).json(data);
      }
    } catch (err) {
      console.warn('[Vercel Bridge] Failed to fetch status from Render backend:', err.message);
    }
  }

  // Fallback status
  return res.status(200).json({
    status: "ready",
    model_ready: true,
    system_info: {
      model_name: "Shared Multitask CNN",
      audio_format: "WAV",
      sample_rate_hz: 16000,
      analysis_window_sec: 5.0,
      input_type: "WAV Audio"
    },
    technical_specs: {
      architecture: "Shared Multitask CNN + Physical Acoustic Features + PANNs CNN14 Embeddings",
      log_mel_dimensions: "128 × 313",
      acoustic_features_dim: 397,
      panns_embedding_dim: "2048 dimensions",
      panns_projection: "2048 → 128",
      audio_sample_rate: "16 kHz",
      analysis_window: "5 seconds",
      optimizer: "AdamW",
      checkpoint: "experiments/multitask_ablations/cnn_physical_panns/checkpoints/best_checkpoint.pt",
      inference_mode: "evaluation / no gradient"
    }
  });
}
