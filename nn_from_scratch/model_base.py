import torch
import torch.nn as nn
#torch.manual_seed(0)

class NeuralNetwork(nn.Module):
  """
  """
  def __init__(self, ):
    super(NeuralNetwork, self).__init__()
    # parameters
    self.input_size = 11
    self.hlayer1_size = 5
    self.hlayer2_size = 6
    self.hlayer3_size = 5
    self.output_size = 10

    # weights
    self.W1 = torch.randn(self.input_size, self.hlayer1_size)    # 11x5
    self.W2 = torch.randn(self.hlayer1_size, self.hlayer2_size)  # 5x6
    self.W3 = torch.randn(self.hlayer2_size, self.hlayer3_size)  # 6x5
    self.W4 = torch.randn(self.hlayer3_size, self.output_size)  # 5x10

  def forward(self, y):
    self.z1 = torch.matmul(y, self.W1)
    self.z2 = self.sigmoid(self.z1)
    self.z3 = torch.matmul(self.z2, self.W2)
    self.z4 = self.sigmoid(self.z3)
    self.z5 = torch.matmul(self.z4, self.W3)
    self.z6 = self.sigmoid(self.z5)
    self.z7 = torch.matmul(self.z6, self.W4)
    x_hat = self.sigmoid(self.z7)
    return x_hat

  def sigmoid(self, s):
    return 1 / (1 + torch.exp(-s))
    
  def sigmoidPrime(self, s):
    return s * (1 - s)

  def backward(self, y, x, x_hat):
    self.x_hat_error = x - x_hat # del C_0/del a(L)
    self.x_hat_delta = self.x_hat_error * self.sigmoidPrime(x_hat) # * del a(L)/del z(L)

    self.z6_error = torch.matmul(self.x_hat_delta, torch.t(self.W4)) # * del z(L)/del a(L-1) 
    self.z6_delta = self.z6_error * self.sigmoidPrime(self.z6) # * del a(L-1)/ del z(L-1)

    self.z4_error = torch.matmul(self.z6_delta, torch.t(self.W3)) # * del z(L-1)/del a(L-2)
    self.z4_delta = self.z4_error * self.sigmoidPrime(self.z4) # * del a(L-2) / del z(L-2)

    self.z2_error = torch.matmul(self.z4_delta, torch.t(self.W2)) # * del z(L-2)/del a(L-3)
    self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2) # * del a(L-3)/del z(L-3)

    self.W1 += torch.matmul(torch.t(y), self.z2_delta)
    self.W2 += torch.matmul(torch.t(self.z2), self.z4_delta)
    self.W3 += torch.matmul(torch.t(self.z4), self.z6_delta)
    self.W4 += torch.matmul(torch.t(self.z6), self.x_hat_delta)

  def train(self, x, y):
    x_hat = self.forward(y)
    self.backward(y, x, x_hat)
