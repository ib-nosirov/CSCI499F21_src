import torch
import torch.nn as nn

import model_fc
import trainer
#import tester

A = torch.randn(10,11)
x_train = torch.randn(100,10)
y_train = torch.matmul(x_train, A)

trainer.train(x_train, y_train, A, 10)
