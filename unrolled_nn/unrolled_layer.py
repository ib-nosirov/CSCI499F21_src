import math
import torch
import torch.nn as nn

class UnrolledLayer(nn.Module):
  """Custom unrolled layer."""
  def __init__(self, y, A):
    super().__init__()
    self.A = A
    self.A_T = torch.transpose(self.A, 0, 1)
    self.eta = nn.Parameter(torch.rand(1), requires_grad=True)
    self.y = y

  def forward(self, x):
    return x - self.eta * torch.matmul(self.A_T, torch.matmul(self.A, x)
           - self.y)
