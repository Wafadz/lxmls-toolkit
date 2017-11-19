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


def log_forward(x, activation_functions, parameters, all_outputs=False):

    num_layers = len(activation_functions)

    # Input
    x = Variable(torch.from_numpy(x).float(), requires_grad=False)

    # Start of the computation graph
    tilde_z = x

    if all_outputs:
        activations = []

    for n in range(num_layers):

        # Get weigths and bias of the layer (even and odd positions)
        W = parameters[2*n]
        b = parameters[2*n+1]

        # Cast variables to Pytorch types
        W = Variable(torch.from_numpy(W).float())
        b = Variable(torch.from_numpy(b).float())

        # Linear transformation
        z = torch.matmul(tilde_z, torch.t(W)) + torch.t(b)

        # Non-linear transformation
        if activation_functions[n] == "sigmoid":
            tilde_z = torch.sigmoid(z)

        elif activation_functions[n] == "softmax":
            # Softmax is computed in log-domain to prevent
            # underflow/overflow
            tilde_z = torch.nn.LogSoftmax()(z)

        else:
            raise Exception("Uknown activation %s" % activation_functions[n])

        if all_outputs:
            activations.append(tilde_z)

    if all_outputs:
        tilde_z = activations

    return tilde_z

test_hat_y = log_forward(test_x.T, mlp.actvfunc, mlp.params, all_outputs=True)
import ipdb;ipdb.set_trace(context=30)
