import torch
import torch.nn as nn

import model_fc
import trainer
#import tester

A = torch.randn(10, 11)
x_train = [torch.randn(11, 5) for i in range(200)]
y_train = [torch.matmul(A, x_train[i]) for i in range(200)]

trainer.train(x_train, y_train, A, 200)
