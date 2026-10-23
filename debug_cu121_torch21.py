import requests
import sys

url = 'https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html'
r = requests.get(url, timeout=10)
print('Status:', r.status_code, flush=True)
print('Length:', len(r.text), flush=True)
print(r.text[:3000], flush=True)