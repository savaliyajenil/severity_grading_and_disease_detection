import requests

for torch_ver in ['torch1.13', 'torch1.12', 'torch1.11', 'torch1.10', 'torch1.9', 'torch1.8', 'torch1.7']:
    url = f'https://download.openmmlab.com/mmcv/dist/cu118/{torch_ver}/index.html'
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200 and len(r.text) > 1000:
            print(f'{torch_ver}: 200 - {len(r.text)} bytes')
        else:
            print(f'{torch_ver}: {r.status_code}')
    except:
        print(f'{torch_ver}: error')