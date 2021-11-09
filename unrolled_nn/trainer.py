"""A function that trains a neural network."""
import model_fc
import torch
import torch.nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.optim as optim

def train(data_train, A, max_epochs, batch_size):
  model = model_fc.ModelFc(A)
  data_loader = DataLoader(data_train, batch_size, shuffle=True)
  loss = torch.nn.MSELoss() 
  optim = torch.optim.SGD(model.parameters(), lr=1e-4)
  for epoch in range(max_epochs):
    for x, y in data_loader:
      optim.zero_grad()
      x_pred = model(x, y)
      loss_val = loss(x_pred, x)
      loss_val.backward()
      optim.step()

    print('epoch = ', epoch, ', train_loss = ', loss_val)
