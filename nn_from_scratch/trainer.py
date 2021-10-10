import model_fc
import torch
import torch.nn as nn

def train(y, x, weights_list, epochs):
  """A function that trains a network on a dataset."""
  for i in range(epochs):
    x_hat, z_list = forward(y, weights_list)
    backward(y, x, x_hat, z_list, weights_list)
    print ("#" + str(i) + " Loss: "
    + str(torch.mean((x - x_hat)**2).detach().item()))

def forward(y, weights_list):
  z_list = [torch.matmul(y, weights_list[0])]
  z = z_list[0]
  z = sigmoid(z)
  z_list.append(z)
  for i in range(1, len(weights_list)):
    z = torch.matmul(z, weights_list[i])
    z_list.append(z)
    z = sigmoid(z)
    z_list.append(z)

  return z, z_list

def sigmoid(s):
  return 1/(1 + torch.exp(-s))

def sigmoidPrime(s):
  return s*(1 - s)

def backward(y, x, x_hat, z_list, weights_list):
  x_hat_error = x - x_hat
  x_hat_delta = x_hat_error * sigmoidPrime(x_hat)

  z6_error = torch.matmul(x_hat_delta, torch.t(weights_list[3]))
  z6_delta = z6_error * sigmoidPrime(z_list[5])

  z4_error = torch.matmul(z6_delta, torch.t(weights_list[2]))
  z4_delta = z4_error * sigmoidPrime(z_list[3])

  z2_error = torch.matmul(z4_delta, torch.t(weights_list[1]))
  z2_delta = z2_error * sigmoidPrime(z_list[1])

  weights_list[0] += torch.matmul(torch.t(y), z2_delta)
  weights_list[1] += torch.matmul(torch.t(z_list[1]), z4_delta)
  weights_list[2] += torch.matmul(torch.t(z_list[3]), z6_delta)
  weights_list[3] += torch.matmul(torch.t(z_list[5]), x_hat_delta)
