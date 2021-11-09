import math
import torch
import torch.nn as nn

class UnrolledLayer(nn.Module):
  """Custom unrolled layer."""
  def __init__(self, A):
    super().__init__()
    self.A = A
    self.A_T = A.transpose(1, 0)
    self.eta = nn.Parameter(torch.rand(1))

  def forward(self, x, y):
    # y has size n_batch x n_measurements
    eta = torch.clamp(self.eta, min=0, max=100)
    x = x.permute(1, 0)
    Ax = self.A.matmul(x)
    y = y.permute(1,0)
    # size n_measurements x n_batch
    res = Ax - y 
    x_out = x - eta * self.A_T.matmul(res)
    return x_out.permute(1, 0)
