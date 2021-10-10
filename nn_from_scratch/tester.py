import torch
import torch.nn as nn

def test(x_test, y_test):
  for i in range(len(x_test)):
    print("Test Loss: " + str(torch.mean((x_test[i] -
    y_test[i])**2).detach().item()))
