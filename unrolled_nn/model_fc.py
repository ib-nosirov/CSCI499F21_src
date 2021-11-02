import model_base
import torch
import torch.nn as nn

class ModelFc(model_base.ModelBase):
  """A fully-connected network with user-specifiable hyperparameters."""

  def __init__(self, y, A):
    super().__init__()

    self.x_k1 = super().layer(y, A)
    self.x_k2 = super().layer(y, A)
    self.x_k3 = super().layer(y, A)
    
    # Build the network layer by layer
  def forward(self, x):
    return self.x_k3(self.x_k2(self.x_k1(x)))

  def set_new_y(self, y):
    self.x_k1.set_y(y)
    self.x_k2.set_y(y)
    self.x_k3.set_y(y)
