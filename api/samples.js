// =============================================================================
// [LAYER: VERCEL SERVERLESS BRIDGE] | File: api/samples.js
// Purpose: Cloud Serverless Sample Audio Metadata Endpoint (/api/samples)
// =============================================================================

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  if (req.method === 'OPTIONS') return res.status(200).end();

  const backendUrl = process.env.EXTERNAL_BACKEND_URL || process.env.RENDER_BACKEND_URL || 'https://ai-stethoscope-backend-production.up.railway.app';
  if (backendUrl) {
    try {
      const targetUrl = `${backendUrl.replace(/\/$/, '')}/api/samples`;
      const response = await fetch(targetUrl, {
        headers: { 'Accept': 'application/json' },
      });
      if (response.ok) {
        const data = await response.json();
        return res.status(200).json(data);
      }
    } catch (err) {
      console.warn('[Vercel Bridge] Failed to fetch samples from Render backend:', err.message);
    }
  }

  // Fallback sample
  return res.status(200).json({
    success: true,
    samples: [
      {
        id: "b0032",
        name: "b0032.wav (Heart Sound - PhysioNet Challenge)",
        path: "datasets/Heart/Raw/training/training-b/b0032.wav",
        filename: "b0032.wav",
        size_bytes: 160044,
        description: "5.0-second cardiac recording from training-b set"
      }
    ]
  });
}
