from __future__ import division
import numpy as np
import torch
from torch.autograd import Variable
from lxmls.deep_learning.rnn import RNN


def cast_float(variable):
    return Variable(torch.from_numpy(variable).float(), requires_grad=True)


def cast_int(variable):
    return Variable(torch.from_numpy(variable).long(), requires_grad=True)


def _log_forward(input, parameters):
    """
    Forward pass
    """

    # Ensure the type matches torch type
    input = Variable(torch.from_numpy(input).long())

    # Get parameters and sizes
    W_e, W_x, W_h, W_y = parameters
    embedding_size, vocabulary_size = W_e.data.shape
    hidden_size = W_h.data.shape[0]
    nr_steps = input.data.shape[0]

    # Define some operations a priori
    # Initialize Embedding layer
    embedding_layer = torch.nn.Embedding(vocabulary_size, embedding_size)
    embedding_layer.weight.data = torch.t(W_e).data
    # Log softmax
    logsoftmax = torch.nn.LogSoftmax()

    # FORWARD PASS COMPUTATION GRAPH

    # Word Embeddings
    z_e = torch.t(embedding_layer(input))

    # Recurrent layer
    h = Variable(torch.FloatTensor(hidden_size, nr_steps + 1).zero_())
    for t in range(nr_steps):

        # Linear
        z_t = torch.matmul(z_e[:, t], torch.t(W_x)) + \
            torch.matmul(h[:, t], torch.t(W_h))

        # Non-linear (sigmoid)
        h[:, t+1] = torch.sigmoid(z_t)

    # Output layer
    y = torch.matmul(W_y, h[:, 1:])

    # Log-Softmax
    log_p_y = torch.t(logsoftmax(torch.t(y)))

    return log_p_y


def backpropagation(input, output, parameters):
    """
    Computes the gradients of the network with respect to cross entropy
    error cost
    """
    output = Variable(
        torch.from_numpy(output).long(),
        requires_grad=False
    )

    # Compute negative log-likelihood loss
    loss = torch.nn.NLLLoss()(_log_forward(input), output)
    # Use autograd to compute the backward pass.
    loss.backward()

    gradient_parameters = []
    for parameter in parameters:
        gradient_parameters.append(parameters.grad.data)
    return gradient_parameters


class PytorchRNN(RNN):
    """
    Basic RNN with forward-pass and gradient computation in Pytorch
    """

    def __init__(self, **config):

        # This will initialize
        # self.num_layers
        # self.config
        # self.parameters
        RNN.__init__(self, **config)

        # Need to cast all weights
        for n in range(len(self.parameters)):
            # Get weigths and bias of the layer (even and odd positions)
            self.parameters[n] = cast_float(self.parameters[n])

    def predict(self, input=None):
        """
        Predict model outputs given input
        """
        p_y = np.exp(self._log_forward(input).data.numpy())
        return np.argmax(p_y, axis=1)

    def update(self, input=None, output=None):
        """
        Update model parameters given batch of data
        """
        gradients = self.gradients(input, output)
        learning_rate = self.config['learning_rate']
        # Update each parameter with SGD rule
        for m in np.arange(self.num_layers):
            # Update weight
            self.parameters[m][0].data -= learning_rate * gradients[m][0]
            # Update bias
            self.parameters[m][1].data -= learning_rate * gradients[m][1]

        # Zero gradients
        for n in np.arange(self.num_layers):
            weight, bias = self.parameters[n]
            weight.grad.data.zero_()
            bias.grad.data.zero_()
