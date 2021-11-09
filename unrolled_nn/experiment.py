import torch
import torch.nn as nn

import model_fc
import trainer
#import tester
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.optim as optim

n_features = 10
n_measurements = 5
n_samples = 1000
max_epochs = 100
batch_size = 100

A = torch.randn(n_measurements, n_features)
x_train = torch.randn(n_samples, n_features)
y_train = A.matmul(x_train.permute(1,0)).permute(1,0)

x_validate = torch.randn(n_samples, n_features)
y_validate = A.matmul(x_train.permute(1,0)).permute(1,0)

data_train = TensorDataset(x_train, y_train)
data_validate = TensorDataset(x_validate, y_validate)
trainerClass = trainer.Trainer(data_train, data_validate, A, max_epochs,
                               batch_size)
trainerClass.train()

#x_test = torch.randn(n_samples, n_features)
#y_test = A.matmul(x_test.permute(1,0)).permute(1,0)
