import requests
import re
import sys

url = 'https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html'
r = requests.get(url, timeout=10)
print('Status:', r.status_code, flush=True)
files = re.findall(r'href=["\'](mmcv[^"\']+)["\']', r.text)
print('Files found:', len(files), flush=True)
for f in files[:10]:
    sys.stdout.write(f + '\n')
    sys.stdout.flush()