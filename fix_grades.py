with open('scripts/11_train_samtseg.py', 'r') as f:
    content = f.read()

content = content.replace(
    'ratios = batch["ratio"].to(device, non_blocking=True)',
    'ratios = batch["ratio"].to(device, non_blocking=True)\n            grades = batch["grade"].to(device, non_blocking=True)'
)

with open('scripts/11_train_samtseg.py', 'w') as f:
    f.write(content)
print('Fixed')