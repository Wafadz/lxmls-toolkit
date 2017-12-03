import numpy as np
from scipy.misc import logsumexp
from lxmls.deep_learning.rnn import RNN
from lxmls.deep_learning.utils import index2onehot


def log_forward(input, parameters):

    # Get parameters and sizes
    W_e, W_x, W_h, W_y = parameters
    hidden_size = W_h.shape[0]
    nr_steps = input.shape[0]

    # Embedding layer
    z_e = W_e[:, input]

    # Recurrent layer
    h = np.zeros((hidden_size, nr_steps+1))
    for t in range(nr_steps):

        # Linear
        z_t = W_x.dot(z_e[:, t]) + W_h.dot(h[:, t])

        # Non-linear
        h[:, t+1] = 1.0 / (1 + np.exp(-z_t))

    # Output layer
    y = W_y.dot(h[:, 1:])

    # Softmax
    log_p_y = y - logsumexp(y, axis=0)

    return log_p_y, y, h, z_e, input


def backpropagation(input, output, parameters):

    '''
    Compute gradientes, with the back-propagation method
    inputs:
        x: vector with the (embedding) indicies of the words of a
            sentence
        outputs: vector with the indicies of the tags for each word of
                    the sentence outputs:
        gradient_parameters: vector with parameters gradientes
    '''

    # Get parameters and sizes
    W_e, W_x, W_h, W_y = parameters
    nr_steps = input.shape[0]

    log_p_y, y, h, z_e, x = log_forward(input, parameters)
    p_y = np.exp(log_p_y)

    # Initialize gradients with zero entrances
    gradient_W_e = np.zeros(W_e.shape)
    gradient_W_x = np.zeros(W_x.shape)
    gradient_W_h = np.zeros(W_h.shape)
    gradient_W_y = np.zeros(W_y.shape)

    # Gradient of the cost with respect to the last linear model
    I = index2onehot(output, W_y.shape[0])
    error = (p_y - I.T) / nr_steps

    # backward pass, with gradient computation
    error_h_next = np.zeros_like(h[:, 0])
    for t in reversed(range(nr_steps)):

        # Output linear
        error_h = np.dot(W_y.T, error[:, t]) + error_h_next

        # Non-linear
        error_raw = h[:, t+1] * (1. - h[:, t+1]) * error_h

        # Hidden-linear
        error_h_next = np.dot(W_h.T, error_raw)

        # Weight gradients
        gradient_W_y += np.outer(error[:, t], h[:, t+1])
        gradient_W_h += np.outer(error_raw, h[:, t])
        gradient_W_x += np.outer(error_raw, z_e[:, t])
        gradient_W_e[:, x[t]] += W_x.T.dot(error_raw)

    # Normalize over sentence length
    gradient_parameters = [
        gradient_W_e, gradient_W_x, gradient_W_h, gradient_W_y
    ]

    return gradient_parameters


def cross_entropy_loss(input, output, parameters):
    """Cross entropy loss"""
    nr_steps = input.shape[0]
    log_probability = log_forward(input, parameters)[0]
    return -log_probability[range(nr_steps), output].mean()


class NumpyRNN(RNN):

    def __init__(self, **config):
        # This will initialize
        # self.config
        # self.parameters
        RNN.__init__(self, **config)

    def predict(self, input=None):
        """
        Predict model outputs given input
        """
        log_forward = self.log_forward(input)[0]
        return np.argmax(np.exp(log_forward), axis=0)

    def update(self, input=None, output=None):
        """
        Update model parameters given batch of data
        """
        gradients = backpropagation(input, output)
        learning_rate = self.config['learning_rate']
        # Update each parameter with SGD rule
        num_parameters = len(self.parameters)
        for m in range(num_parameters):
            # Update weight
            self.parameters[m][0] -= learning_rate * gradients[m][0]
            # Update bias
            self.parameters[m][1] -= learning_rate * gradients[m][1]
