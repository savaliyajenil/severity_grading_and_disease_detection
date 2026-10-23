import requests
import re

for ver in ['cu121', 'cu118', 'cu117']:
    for torch_ver in ['torch2.3', 'torch2.2', 'torch2.1', 'torch2.0', 'torch1.13', 'torch1.12', 'torch1.11', 'torch1.10', 'torch1.9', 'torch1.8', 'torch1.7']:
        url = f'https://download.openmmlab.com/mmcv/dist/{ver}/{torch_ver}/index.html'
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200 and len(r.text) > 1000:
                files = re.findall(r'href=["\'](mmcv-full[^"\']+)["\']', r.text)
                if files:
                    print(f'{ver}/{torch_ver}: {files[:3]}')
        except:
            pass