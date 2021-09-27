if __name__ == "__main__":
  # Create the dataset for Ax = y
  A = torch.randn(10,11)
  x_train = torch.randn(100,10)
  y_train = torch.matmul(x_train, A)

  # We will use values of y to predict x, in effect modelling the linear
  # transformation A.
  NN = NeuralNetwork()

  for i in range(1000):  # trains the NN 1,000 times
    print ("#" + str(i) + " Loss: "
    + str(torch.mean((x_train - NN.forward(y_train))**2).detach().item()))
    NN.train(x_train, y_train)

  x_test = [torch.randn(1,10) for i in range(10)]
  y_test = [torch.matmul(x_test[i], A) for i in range(10)]

  for i in range(10):
    print ("Test Loss: "
    + str(torch.mean((x_test[i] - NN.forward(y_test[i]))**2).detach().item()))
