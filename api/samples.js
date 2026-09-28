export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.status(200).end();

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
