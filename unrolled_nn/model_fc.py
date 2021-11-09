import model_base
import torch
import torch.nn as nn

class ModelFc(model_base.ModelBase):
  """A fully-connected network with user-specifiable hyperparameters."""

  def __init__(self, A):
    super().__init__()

    self.x_k1 = super().unrolledLayer(A)
    self.x_k2 = super().unrolledLayer(A)
    self.x_k3 = super().unrolledLayer(A)
    
    # Build the network layer by layer
  def forward(self, x, y):
    return self.x_k3(self.x_k2(self.x_k1(x, y), y), y)

