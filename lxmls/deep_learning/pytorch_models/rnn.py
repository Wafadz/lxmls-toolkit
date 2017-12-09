from __future__ import division
import numpy as np
import torch
from torch.autograd import Variable
from lxmls.deep_learning.rnn import RNN


def cast_float(variable):
    return Variable(torch.from_numpy(variable).float(), requires_grad=True)


def cast_int(variable):
    return Variable(torch.from_numpy(variable).long(), requires_grad=True)


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

        # First parameters are the embeddings
        # instantiate the embedding layer first
        self.embedding_layer = torch.nn.Embedding(
            config['vocabulary_size'],
            config['embedding_size']
        )
        # Set its value to the stored weight
        self.embedding_layer.weight.data = \
            torch.from_numpy(self.parameters[0]).float()
        # Store the pytorch variable in our parameter list
        self.parameters[0] = self.embedding_layer.weight

        # Need to cast  rest of weights
        num_parameters = len(self.parameters)
        for index in range(1, num_parameters):
            # Get weigths and bias of the layer (even and odd positions)
            self.parameters[index] = cast_float(self.parameters[index])

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
        gradients = self.backpropagation(input, output)
        learning_rate = self.config['learning_rate']
        # Update each parameter with SGD rule
        num_parameters = len(self.parameters)
        for m in np.arange(num_parameters):
            # Update weight
            import ipdb;ipdb.set_trace(context=30)
            self.parameters[m].data -= learning_rate * gradients[m]

    def _log_forward(self, input):
        """
        Forward pass
        """

        # Ensure the type matches torch type
        input = Variable(torch.from_numpy(input).long())

        # Get parameters and sizes
        W_e, W_x, W_h, W_y = self.parameters
        embedding_size, vocabulary_size = W_e.shape
        hidden_size = W_h.shape[0]
        nr_steps = input.shape[0]

        # Define some operations a priori
        # Initialize Embedding layer
        # Log softmax
        logsoftmax = torch.nn.LogSoftmax(dim=0)

        # FORWARD PASS COMPUTATION GRAPH

        # Word Embeddings
        z_e = torch.t(self.embedding_layer(input))

        # Recurrent layer
        h = Variable(torch.FloatTensor(hidden_size, nr_steps + 1).zero_())
        for t in range(nr_steps):

            # Linear
            z_t = torch.matmul(
                z_e[:, t].clone(),
                torch.t(W_x)
            ) + torch.matmul(
                h[:, t].clone(),
                torch.t(W_h)
            )

            # Non-linear (sigmoid)
            h[:, t+1] = torch.sigmoid(z_t)

        # Output layer
        y = torch.matmul(W_y, h[:, 1:])

        # Log-Softmax
        log_p_y = logsoftmax(y)

        return log_p_y

    def backpropagation(self, input, output):
        """
        Computes the gradients of the network with respect to cross entropy
        error cost
        """
        output = Variable(
            torch.from_numpy(output).long(),
            requires_grad=False
        )

        loss_function = torch.nn.NLLLoss()

        # Zero gradients
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad.data.zero_()

        # Compute negative log-likelihood loss
        log_p_y = self._log_forward(input)
        cost = loss_function(torch.t(log_p_y), output)
        # Use autograd to compute the backward pass.
        cost.backward()

        num_parameters = len(self.parameters)
        gradient_parameters = [torch.t(self.parameters[0].grad.data)]
        for index in range(1, num_parameters):
            gradient_parameters.append(self.parameters[index].grad.data)

        return gradient_parameters
