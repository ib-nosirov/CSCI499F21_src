"""A function that trains a neural network."""
import model_fc
import torch
import torch.nn

def train(x, y, A, epochs):
  model = model_fc.ModelFc(y[0], A)
  for i in range(epochs):
    for j in range(2):
      x_k = model(x[j])
      loss = 0.5*((torch.linalg.norm(x_k))**2)
      loss.backward()
      optim = torch.optim.SGD(model.parameters(), lr=1e-2)
      optim.step()
      print(list(model.parameters()))
#      if(j%100 == 0):
#        print(loss)
