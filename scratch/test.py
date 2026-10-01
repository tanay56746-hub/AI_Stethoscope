import urllib.request
import time

filepath = 'datasets/Heart/Raw/training/training-b/b0032.wav'
with open(filepath, 'rb') as f:
    audio_data = f.read()

boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
body = (
    b'--' + boundary.encode() + b'\r\n'
    b'Content-Disposition: form-data; name="audio"; filename="b0032.wav"\r\n'
    b'Content-Type: audio/wav\r\n\r\n' +
    audio_data + b'\r\n'
    b'--' + boundary.encode() + b'--\r\n'
)

req = urllib.request.Request(
    'https://ai-stethoscope-backend-production.up.railway.app/api/predict',
    data=body,
    headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
)

start = time.time()
try:
    with urllib.request.urlopen(req, timeout=30) as res:
        print('HTTP', res.getcode())
        print(f'Time taken: {time.time() - start:.2f}s')
        print('Success!')
except Exception as e:
    print('Failed:', e)
    print(f'Time taken: {time.time() - start:.2f}s')
