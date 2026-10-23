import requests
import re
import sys

url = 'https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html'
r = requests.get(url, timeout=10)
print('Status:', r.status_code, flush=True)
print('Length:', len(r.text), flush=True)

# Look for mmcv-full
files = re.findall(r'href=["\'](mmcv-full[^"\']+)["\']', r.text)
print('mmcv-full files found:', len(files), flush=True)
for f in files[:10]:
    sys.stdout.write(f + '\n')
    sys.stdout.flush()

# Also check for mmcv (not full)
files2 = re.findall(r'href=["\'](mmcv[^"\']+)["\']', r.text)
print('mmcv files found:', len(files2), flush=True)
for f in files2[:10]:
    sys.stdout.write(f + '\n')
    sys.stdout.flush()