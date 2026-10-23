import requests
import re

url = 'https://download.openmmlab.com/mmcv/dist/index.html'
r = requests.get(url, timeout=10)
cus = re.findall(r'href=["\'](cu[0-9]+)["\']', r.text)
print('Available CUDA versions:', sorted(set(cus)))