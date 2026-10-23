import requests
import re

url = 'https://download.openmmlab.com/mmcv/dist/index.html'
r = requests.get(url, timeout=10)
print('Status:', r.status_code)
print('Length:', len(r.text))

# Find all cuXXX patterns
cus = re.findall(r'cu[0-9]+', r.text)
print('CUDA versions:', sorted(set(cus)))

# Find torch versions
torch_vers = re.findall(r'torch[0-9.]+', r.text)
print('Torch versions:', sorted(set(torch_vers)))

# Find mmcv versions
mmcv_vers = re.findall(r'mmcv[^<>"]+', r.text)
print('MMCV versions:', sorted(set(mmcv_vers))[:20])