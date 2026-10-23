import requests
import re
import sys

url = 'https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html'
r = requests.get(url, timeout=10)
print('Status:', r.status_code, flush=True)
print('Length:', len(r.text), flush=True)
print(r.text[:2000], flush=True)