from __future__ import division
import numpy as np
from scipy.misc import logsumexp
from lxmls.deep_learning.mlp import MLP
from lxmls.deep_learning.utils import index2onehot


class NumpyMLP(MLP):
    """
    Basic MLP with forward-pass and gradient computation in Numpy
    """

    def __init__(self, **config):

        # This will initialize
        # self.num_layers
        # self.config
        # self.parameters
        MLP.__init__(self, **config)

    def log_forward(self, input, all_inputs=False):
        """
        Forward pass

        all_inputs = True  return intermediate activations
        """

        # This will store activations at each layer
        activation_functions = self.config['activation_functions']

        # Input
        tilde_z = input
        layer_inputs = []

        for n in range(self.num_layers):

            # Store input to this layer (needed for backpropagation)
            layer_inputs.append(tilde_z)

            # Get weigths and bias of the layer (even and odd positions)
            weight, bias = self.parameters[n]

            # Linear transformation
            z = np.dot(tilde_z, weight.T) + bias

            # Non-linear transformation
            if activation_functions[n] == "sigmoid":
                tilde_z = 1.0 / (1 + np.exp(-z))

            elif activation_functions[n] == "softmax":
                # Softmax is computed in log-domain to prevent
                # underflow/overflow
                log_tilde_z = z - logsumexp(z, axis=1)[:, None]

        if all_inputs:
            return log_tilde_z, layer_inputs
        else:
            return log_tilde_z

    def gradients(self, input, output):
        """
        Computes the gradients of the network with respect to cross entropy
        error cost
        """

        # Run forward and store activations for each layer
        log_prob_y, layer_inputs = self.log_forward(input, all_inputs=True)
        prob_y = np.exp(log_prob_y)

        # For each layer in reverse store the gradients for each parameter
        activation_functions = self.config['activation_functions']
        nabla_parameters = []
        for n in np.arange(self.num_layers-1, -1, -1):

            # Get weigths and bias (always in even and odd positions)
            # Note that sometimes we need the weight from the next layer
            W = self.parameters[n][0]
            if n != self.num_layers-1:
                W_next = self.parameters[n+1][0]

            # ----------
            # Solution to Exercise 6.2

            # If it is the last layer, compute the average cost gradient
            # Otherwise, propagate the error backwards from the next layer
            if n == self.num_layers-1:
                # NOTE: This assumes cross entropy cost
                if activation_functions[n] == 'sigmoid':
                    e = (prob_y - output) / output.shape[0]
                elif activation_functions[n] == 'softmax':
                    I = index2onehot(output, W.shape[0])
                    e = (prob_y - I) / output.shape[0]

            else:

                # Backpropagate through linear layer
                e = np.dot(e, W_next)

                # Backpropagate through sigmoid layer
                e *= layer_inputs[n+1] * (1-layer_inputs[n+1])

            # Weight gradient
            nabla_W = np.zeros(W.shape)
            for l in np.arange(e.shape[0]):
                nabla_W += np.outer(e[l, :], layer_inputs[n][l, :])
            # Bias gradient
            nabla_b = np.sum(e, axis=0, keepdims=True)

            # End of solution to Exercise 6.2
            # ----------

            # Store the gradients
            nabla_parameters.append((nabla_W, nabla_b))

        return nabla_parameters[::-1]

    def predict(self, input=None):
        """
        Predict model outputs given input
        """
        log_forward = self.log_forward(input)
        return np.argmax(np.exp(log_forward), axis=1)

    def update(self, input=None, output=None):
        """
        Update model parameters given batch of data
        """
        gradients = self.gradients(input, output)
        learning_rate = self.config['learning_rate']
        # Update each parameter with SGD rule
        for m in np.arange(self.num_layers):
            # Update weight
            self.parameters[m][0] -= learning_rate * gradients[m][0]
            # Update bias
            self.parameters[m][1] -= learning_rate * gradients[m][1]
