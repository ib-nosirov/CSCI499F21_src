import model_base
import torch
import torch.nn as nn

class ModelFc(model_base.ModelBase):
  """A fully-connected network with user-specifiable hyperparameters."""

  def __init__(self, y, A):
    super().__init__()

    self.x_k1 = self.layer(y, A)
    self.x_k2 = self.layer(y, A)
    self.x_k3 = self.layer(y, A)
    
    # Build the network layer by layer
  def forward(self, x):
    return self.x_k3(self.x_k2(self.x_k1(x)))
