import numpy as np
import lxmls.readers.sentiment_reader as srs
scr = srs.SentimentCorpus("books")
train_x = scr.train_X.T
train_y = scr.train_y[:, 0]
test_x = scr.test_X.T
test_y = scr.test_y[:, 0]

# Neural network modules
import lxmls.deep_learning.mlp as dl
import lxmls.deep_learning.sgd as sgd
# Model parameters
geometry = [train_x.shape[0], 20, 2]
actvfunc = ['sigmoid', 'softmax']

# Model parameters
n_iter = 5
bsize = 5
lrate = 0.05

# NUMPY PART
mlp = dl.NumpyMLP(geometry, actvfunc)

# PYTORCH PART
import torch
from torch.autograd import Variable
# Weigths and bias of first layer
W1, b1 = mlp.params[0:2]

def forward(x, W1, b1):

    # Cast variables to Pytorch types
    dtype = torch.FloatTensor
    _x = Variable(torch.from_numpy(x).float(), requires_grad=False)
    _W1 = Variable(torch.from_numpy(W1).float())
    _b1 = Variable(torch.from_numpy(b1).float())

    # Computation graph
    _z1 = torch.matmul(_x, torch.t(_W1)) + torch.t(_b1)
    _z2 = torch.sigmoid(_z1)

    return _z2

forward(test_x.T, W1, b1)
import ipdb;ipdb.set_trace(context=30)
