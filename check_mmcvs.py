import requests
import re

url = 'https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html'
r = requests.get(url, timeout=10)
print('Status:', r.status_code)
print('Length:', len(r.text))
files = re.findall(r'href=["\'](mmcv[^"\']+)["\']', r.text)
for f in files[:10]:
    print(f)