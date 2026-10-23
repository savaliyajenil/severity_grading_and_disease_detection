with open(r'C:\Users\student\AppData\Local\Programs\Python\Python310\Lib\site-packages\mmseg\__init__.py', 'r') as f:
    content = f.read()

content = content.replace("MMCV_MAX = '2.2.0'", "MMCV_MAX = '2.3.0'")

with open(r'C:\Users\student\AppData\Local\Programs\Python\Python310\Lib\site-packages\mmseg\__init__.py', 'w') as f:
    f.write(content)

print('Patched')