import torch
import torch.nn as nn
from torch.fx import symbolic_trace

class SplitHardswish(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        return x * self.relu6(x + 3) / 6

model = SplitHardswish()

gm = symbolic_trace(model)

print(gm.graph)
