import torch
import torch.nn as nn

class ModelBase(nn.Module):
  """A base class used for making a neural network."""
  
  def __init__(self, weights=None):
    """Creates dictionaries for storing model weights.
   
    Args:
      weights: a list containing dimensions of input, hidden layers,
      output.
    """
  def dense_layer(self, row_num, col_num):
    """Create weight matrices"""
    return torch.randn(row_num, col_num)
