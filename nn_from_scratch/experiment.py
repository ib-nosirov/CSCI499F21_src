import torch
import torch.nn as nn

import model_fc
import trainer
import tester

A = torch.randn(10,11)
x_train = torch.randn(100,10)
y_train = torch.matmul(x_train, A)

weights = [11, 5, 6, 5, 10]

NN = model_fc.ModelFc(weights)

trainer.train(y_train, x_train, NN.weights_list, 1000)

x_test = [torch.randn(1,10) for i in range(10)]
y_test = [torch.matmul(x_test[i], A) for i in range(10)]
y_test = [trainer.forward(y_test[i], NN.weights_list)[0] for i in range(10)]

tester.test(x_test, y_test)
