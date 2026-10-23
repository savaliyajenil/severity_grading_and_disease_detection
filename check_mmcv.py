import requests
import re

url = 'https://download.openmmlab.com/mmcv/dist/cu121/torch2.3/index.html'
r = requests.get(url, timeout=10)
files = re.findall(r'href=["\'](mmcv[^"\']+)["\']', r.text)
for f in sorted(set(files)):
    print(f)