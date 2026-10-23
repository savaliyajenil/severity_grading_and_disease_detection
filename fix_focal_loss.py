#!/usr/bin/env python3
"""Fix the focal_loss.py file"""

with open('C:/Users/student/Desktop/pdeu_kaam/SegNeXt/mmseg/models/losses/focal_loss.py') as f:
    content = f.read()

# Find the problematic duplicate section and fix it
lines = content.split('\n')

# Find the line with the garbled text
for i, line in enumerate(lines):
    if 'ch.Tensor): The prediction with shape (N, C), C is the number' in line:
        print(f'Found garbled text at line {i+1}')
        # Remove from this line to the next "def " or class definition
        for j in range(i, len(lines)):
            if lines[j].strip().startswith('def ') or lines[j].strip().startswith('class '):
                print(f'Cutting from line {i+1} to {j}')
                lines = lines[:i] + lines[j:]
                break
        break

# Write fixed content
with open('C:/Users/student/Desktop/pdeu_kaam/SegNeXt/mmseg/models/losses/focal_loss.py', 'w') as f:
    f.write('\n'.join(lines))
print('Fixed file written')