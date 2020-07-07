#multioutputregression.py

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import imageio




class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x



def main():
	torch.manual_seed(1)
	num_features = x.shape(1)
	x, y = Variable(x), Variable(y)
	net = Net(n_feature= num_features, n_hidden=10, n_output=1)     # define the network
	print(net)  # net architecture
	optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
	loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss