import model_base
import torch
import torch.nn as nn

class ModelFc(model_base.ModelBase):
  """A fully-connected network with user-specifiable hyperparameters."""
  
  def __init__(self, weights=None):
    """Creates a fully-connected network.
    Args:
      weights: A list of input dimensions for the network.
    """
    # Call parent constructor.
    super(ModelFc, self).__init__(weights=weights)
    
    # Build the network layer by layer.
    self.weights_list = [super().dense_layer(weights[0], weights[1])]
    print(self.weights_list[-1])

    for i in range(2,len(weights)):
      self.weights_list.append(super().dense_layer(weights[i-1],
                                                   weights[i]))
      print(self.weights_list[-1])
