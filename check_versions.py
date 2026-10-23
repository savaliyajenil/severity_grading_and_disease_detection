import requests

for ver in ['cu118', 'cu121']:
    for torch_ver in ['torch2.3', 'torch2.2', 'torch2.1', 'torch2.0']:
        url = f'https://download.openmmlab.com/mmcv/dist/{ver}/{torch_ver}/index.html'
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200 and len(r.text) > 100:
                import re
                files = re.findall(r'href=["\'](mmcv[^"\']+)["\']', r.text)
                if files:
                    print(f'{ver}/{torch_ver}: {files[:3]}')
        except Exception as e:
            pass