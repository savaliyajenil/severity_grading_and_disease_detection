import torch

ckpt = torch.load('C:/Users/student/Desktop/pdeu_kaam/results/segnext_plantseg_exact/checkpoints/epoch_0001.pth', map_location='cpu')
opt = ckpt['optimizer_state_dict']
print('Optimizer keys:', list(opt.keys()))
print('Param groups:')
for i, pg in enumerate(opt['param_groups']):
    print('  Group', i, 'lr=', pg['lr'], 'weight_decay=', pg['weight_decay'])

# Also check scheduler
sched = ckpt.get('scheduler_state_dict')
if sched:
    print('Scheduler:', sched)
else:
    print('No scheduler state dict')

# Check metrics
metrics = ckpt.get('metrics')
print('Metrics:', metrics)