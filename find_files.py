import requests
import re

url = 'https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html'
r = requests.get(url, timeout=10)
print('Status:', r.status_code)
files = re.findall(r'href=["\'](mmcv[^"\']+)["\']', r.text)
for f in files[:10]:
    print(f, flush=True)