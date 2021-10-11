"""A function that trains a neural network."""
import model_fc
import torch
import torch.nn

def train(x, y, A, epochs):
  model = model_fc.ModelFc(y, A)
  for i in range(epochs):
    x_k = model(x)
    loss = 0.5*((torch.linalg.norm(x_k))**2)
    print(loss)
    loss.backward()
    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    optim.step()
