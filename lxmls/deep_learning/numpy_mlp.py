from __future__ import division
import numpy as np
from scipy.misc import logsumexp
#
from lxmls.deep_learning.mlp import MLP, index2onehot


class NumpyMLP(MLP):
    """
    Basic MLP with forward-pass and gradient computation in Numpy
    """

    def __init__(self, geometry, actvfunc, model_file=None):
        """
        Input: geometry  tuple with sizes of layer

        Input: actvfunc  list of strings indicating the type of activation
                         function. Supported 'sigmoid', 'softmax'

        Input: rng       string inidcating random seed
        """

        # CHECK THE PARAMETERS ARE VALID
        self.sanity_checks(geometry, actvfunc, model_file)

        # THIS DEFINES THE MLP
        self.n_layers = len(actvfunc)
        if model_file:
            # Load model
            self.params, self.actvfunc = self.load(model_file)
        else:
            # Parameters are stored as [weight0, bias0, weight1, bias1, ... ]
            # for consistency with the theano way of storing parameters
            self.params, self.actvfunc = self.init_weights(geometry, actvfunc)

    def forward(self, x, all_inputs=False):
        """
        Forward pass

        all_inputs = True  return intermediate activations
        """

        # This will store activations at each layer and the input. This is
        # needed to compute backpropagation
        if all_inputs:
            activations = []

        # Input
        tilde_z = x

        for n in range(self.n_layers):

            # Get weigths and bias of the layer (even and odd positions)
            W = self.params[2*n]
            b = self.params[2*n+1]

            # Linear transformation
            z = np.dot(tilde_z, W.T) + b

            # Non-linear transformation
            if self.actvfunc[n] == "sigmoid":
                tilde_z = 1.0 / (1 + np.exp(-z))

            elif self.actvfunc[n] == "softmax":
                # Softmax is computed in log-domain to prevent
                # underflow/overflow
                tilde_z = np.exp(z - logsumexp(z, axis=1)[:, None])

            if all_inputs:
                activations.append(tilde_z)

        if all_inputs:
            tilde_z = activations

        return tilde_z

    def grads(self, x, y):
        """
       Computes the gradients of the network with respect to cross entropy
       error cost
       """

        # Run forward and store activations for each layer
        layer_inputs = self.forward(x, all_inputs=True)

        # For each layer in reverse store the gradients for each parameter
        nabla_params = [None] * (2*self.n_layers)

        for n in np.arange(self.n_layers-1, -1, -1):

            # Get weigths and bias (always in even and odd positions)
            # Note that sometimes we need the weight from the next layer
            W = self.params[2*n]
            if n != self.n_layers-1:
                W_next = self.params[2*(n+1)]

            # ----------
            # Solution to Exercise 6.2

            # If it is the last layer, compute the average cost gradient
            # Otherwise, propagate the error backwards from the next layer
            if n == self.n_layers-1:
                # NOTE: This assumes cross entropy cost
                if self.actvfunc[n] == 'sigmoid':
                    e = (layer_inputs[n] - y) / y.shape[0]
                elif self.actvfunc[n] == 'softmax':
                    I = index2onehot(y, W.shape[0])
                    e = (layer_inputs[n] - I) / y.shape[0]

            else:

                # Backpropagate through linear layer
                e = np.dot(e, W_next)

                # Backpropagate through sigmoid layer
                # This is correct but confusing n+1 is n in the guide
                e *= layer_inputs[n] * (1-layer_inputs[n])

            # Weight gradient
            nabla_W = np.zeros(W.shape)
            for l in np.arange(e.shape[0]):
                if n == 0:
                    # For the first layer, the activation is the input
                    nabla_W += np.outer(e[l, :], x[l, :])
                else:
                    nabla_W += np.outer(e[l, :], layer_inputs[n-1][l, :])
            # Bias gradient
            nabla_b = np.sum(e, axis=0, keepdims=True)

            # End of solution to Exercise 6.2
            # ----------

            # Store the gradients
            nabla_params[2*n] = nabla_W
            nabla_params[2*n+1] = nabla_b

        return nabla_params
