import math
import torch
import torch.nn as nn

class UnrolledLayer(nn.Module):
  """Custom unrolled layer."""
  def __init__(self, y, A):
    super().__init__()
    self.A = A
    self.A_T = torch.transpose(self.A, 0, 1)
    self.eta = nn.Parameter(torch.rand(1))
    self.y = y

  def forward(self, x):
    eta_A_T = torch.mul(self.A_T, self.eta)
    A_x = torch.matmul(x, self.A)
    return x - torch.matmul(A_x - self.y, eta_A_T)
