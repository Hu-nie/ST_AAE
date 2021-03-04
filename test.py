import torch
import torch.nn as nn


x = torch.rand(3,256,16000)
print('input_size:', x.shape)
conv1d = nn.Conv1d(256, 128, kernel_size=1)
print('kernel_size:', conv1d.weight.shape)
out = conv1d(x)
print('output_size:',out.shape)