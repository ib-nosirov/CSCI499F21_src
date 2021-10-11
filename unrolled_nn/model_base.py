import torch
import torch.nn as nn

import unrolled_layer as custom

class ModelBase(nn.Module):
  def __init__(self):
    super().__init__()

  def layer(self, y, A):
    return custom.UnrolledLayer(y, A)
