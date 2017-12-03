import numpy as np
from scipy.misc import logsumexp
from lxmls.deep_learning.mlp import MLP
from lxmls.deep_learning.utils import index2onehot


def log_forward(input, parameters):
    """Forward pass for sigmoid hidden layers and output softmax"""

    # Input
    tilde_z = input
    layer_inputs = []

    # Hidden layers
    num_hidden_layers = len(parameters) - 1
    for n in range(num_hidden_layers):

        # Store input to this layer (needed for backpropagation)
        layer_inputs.append(tilde_z)

        # Linear transformation
        weight, bias = parameters[n]
        z = np.dot(tilde_z, weight.T) + bias

        # Non-linear transformation (sigmoid)
        tilde_z = 1.0 / (1 + np.exp(-z))

    # Store input to this layer (needed for backpropagation)
    layer_inputs.append(tilde_z)

    # Output linear transformation
    weight, bias = parameters[num_hidden_layers]
    z = np.dot(tilde_z, weight.T) + bias

    # Softmax is computed in log-domain to prevent underflow/overflow
    log_tilde_z = z - logsumexp(z, axis=1)[:, None]

    return log_tilde_z, layer_inputs


def backpropagation(input, output, parameters):
    """Gradients for sigmoid hidden layers and output softmax"""

    # Run forward and store activations for each layer
    log_prob_y, layer_inputs = log_forward(input, parameters)
    prob_y = np.exp(log_prob_y)

    num_examples, num_clases = prob_y.shape
    num_hidden_layers = len(parameters) - 1

    # For each layer in reverse store the backpropagated error, then compute
    # the gradients from the errors and the layer inputs
    errors = []

    # ----------
    # Solution to Exercise 6.2

    # Initial error is the cost derivative at the last layer (for cross
    # entropy cost)
    I = index2onehot(output, num_clases)
    error = (prob_y - I) / num_examples
    errors.append(error)

    # Backpropagate through each layer
    for n in reversed(range(num_hidden_layers)):

        # Note that we only need the weight from the next layer
        weight_next, _ = parameters[n+1]

        # Backpropagate through linear layer
        error = np.dot(error, weight_next)

        # Backpropagate through sigmoid layer
        error *= layer_inputs[n+1] * (1-layer_inputs[n+1])

        # Collect error
        errors.append(error)

    # Reverse errors
    errors = errors[::-1]

    # Compute gradients from errors
    gradients = []
    for n in range(num_hidden_layers + 1):

        weight, _ = parameters[n]

        # Weight gradient
        weight_gradient = np.zeros(weight.shape)
        for l in range(num_examples):
            weight_gradient += np.outer(errors[n][l, :], layer_inputs[n][l, :])

        # Bias gradient
        bias_gradient = np.sum(errors[n], axis=0, keepdims=True)

        # Store gradients
        gradients.append([weight_gradient, bias_gradient])

    # End of solution to Exercise 6.2
    # ----------

    return gradients


def cross_entropy_loss(input, output, parameters):
    """Cross entropy loss"""
    num_examples = input.shape[0]
    log_probability, _ = log_forward(input, parameters)
    return -log_probability[range(num_examples), output].mean()


class NumpyMLP(MLP):
    """
    Basic MLP with forward-pass and gradient computation in Numpy
    """

    def __init__(self, **config):

        # This will initialize
        # self.config
        # self.parameters
        MLP.__init__(self, **config)

    def predict(self, input=None):
        """
        Predict model outputs given input
        """
        log_class_probabilities, _ = log_forward(input, self.parameters)
        return np.argmax(np.exp(log_class_probabilities), axis=1)

    def update(self, input=None, output=None):
        """
        Update model parameters given batch of data
        """

        gradients = backpropagation(input, output, self.parameters)

        learning_rate = self.config['learning_rate']
        num_parameters = len(self.parameters)
        for m in np.arange(num_parameters):

            # Update weight
            self.parameters[m][0] -= learning_rate * gradients[m][0]

            # Update bias
            self.parameters[m][1] -= learning_rate * gradients[m][1]
