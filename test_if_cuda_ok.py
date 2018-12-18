import torch
if torch.cuda.is_available():
    print('cuda acailable')
else:
    print('cuda not acailable')
