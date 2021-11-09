"""A function that trains a neural network."""
import model_fc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.optim as optim

class Trainer(nn.Module):
  """Class for training and validating neural network."""
  def __init__(self, data_train, data_validate, A, max_epochs, batch_size):
    super(Trainer, self).__init__()
    self.train_set = DataLoader(data_train, batch_size, shuffle=True)
    self.validate_set = DataLoader(data_validate, batch_size, shuffle=True)
    self.A = A
    self.max_epochs = max_epochs
    self.batch_size = batch_size
    self.model = None
    self.loss = None
    self.optim = None

  def train_init(self):
    self.model = model_fc.ModelFc(self.A)
    self.loss = torch.nn.MSELoss() 
    self.optim = torch.optim.SGD(self.model.parameters(), lr=1e-4)

  def learn(self, loss_val):
      self.optim.zero_grad()
      loss_val.backward()
      self.optim.step()

  def createTrainCost(self):
    running_loss = 0
    for x, y in self.train_set:
      x_pred = self.model(x, y)
      loss_val = self.loss(x_pred, x)
      running_loss += loss_val
      self.learn(loss_val)
    return running_loss / len(self.validate_set)

  def createValidateCost(self):
    running_loss = 0
    for x, y in self.validate_set:
      x_pred = self.model(x, y)
      running_loss += self.loss(x_pred, x)
    return running_loss / len(self.validate_set)

  def train(self):
    self.train_init()
    for epoch in range(self.max_epochs):
      train_cost = self.createTrainCost()
      validate_cost = self.createValidateCost()
      self.model.writeTrainingSummaries(train_cost, validate_cost)
  
      print('epoch = #', epoch, ', train_cost = ', train_cost, 
            ', validate_cost = ', validate_cost)
