import numpy as np
from scipy.misc import logsumexp
from lxmls.deep_learning.utils import index2onehot


class NumpyRNN():

    def __init__(self, n_words, n_emb, n_hidd, n_tags, seed=None):
        '''
        n_words int         Size of the vocabulary
        n_emb   int         Size of the embeddings
        n_hidd  int         Size of the recurrent layer
        n_tags  int         Total number of tags
        seed    int         Seed to random initialization of parameters (default=None)
        '''

        # MODEL PARAMETERS
        if not seed:
            np.random.seed(0)
        else:
            np.random.seed(seed)

        W_e = 0.01*np.random.uniform(size=(n_emb, n_words))   # Input layer
        W_x = np.random.uniform(size=(n_hidd, n_emb))   # Input layer
        W_h = np.random.uniform(size=(n_hidd, n_hidd))  # Recurrent layer
        W_y = np.random.uniform(size=(n_tags, n_hidd))  # Output layer

        # Class variables
        self.n_hidd = n_hidd
        self.param  = [W_e, W_x, W_h, W_y]
        #self.param_names  = ['W_e', 'W_x', 'W_h', 'W_y']
        self.activation_function = 'logistic'   # 'tanh' 'relu' 'logistic'

    def apply_activation(self, x, function_name):
        '''
        '''
        if function_name == 'logistic':
            z = 1 / (1 + np.exp(-x))
        elif function_name == 'tanh':
            z = np.tanh(x)
        elif function_name == 'relu':
            z = x
            ind = np.where(z < 0.)
            z[ind] = 0.
        else:
            raise NotImplementedError("Unknown activation %s" % function_name)
        return z

    def derivate_activation(self, z, function_name):
        '''
        '''
        if function_name == 'logistic':
            dx = z * (1. - z)
        elif function_name == 'tanh':
            dx = (1. - z * z)
        elif function_name == 'relu':
            dx = (np.sign(z)+1)/2.
        else:
            raise NotImplementedError("Unknown activation %s" % function_name)
        return dx

    def soft_max(self, x, alpha=1.0):
        '''
        '''
        e = np.exp(x / alpha)
        return e / np.sum(e)

    def forward(self, x, all_outputs=False):
        '''
        Forward pass

        all_outputs = True  return intermediate activations; needed to comput
                            backpropagation
        '''
        # Get parameters in nice form
        W_e, W_x, W_h, W_y = self.param

        nr_steps = x.shape[0]
        embbeding_size = W_e.shape[0]
        hidden_size = W_h.shape[0]
        nr_tags = W_y.shape[0]

        # Embedding layer
        z = W_e[:, x]

        # Recursive layer
        h = np.zeros((self.n_hidd, nr_steps+1))
        for t in xrange(nr_steps):
            h[:, t+1] = self.apply_activation(W_x.dot(z[:, t])
                                              + W_h.dot(h[:, t]),
                                              self.activation_function)

        # Output layer
        y = W_y.dot(h[:, 1:])
        p_y = np.exp(y - logsumexp(y, 0))

        if all_outputs:
            return p_y, y, h, z, x
        else:
            return p_y

    def grads(self, x, outputs):
        '''
            Compute gradientes, with the back-propagation method
            inputs:
                x: vector with the (embedding) indicies of the words of a sentence
                outputs: vector with the indicies of the tags for each word of the sentence
            outputs:
                nabla_params: vector with parameters gradientes
        '''

        # Get parameters
        W_e, W_x, W_h, W_y = self.param
        nr_steps = x.shape[0]

        p_y, y, h, z, x = self.forward(x, all_outputs=True)

        # Initialize gradients with zero entrances
        nabla_W_e = np.zeros(W_e.shape)
        nabla_W_x = np.zeros(W_x.shape)
        nabla_W_h = np.zeros(W_h.shape)
        nabla_W_y = np.zeros(W_y.shape)

        # Gradient of the cost with respect to the last linear model
        I = index2onehot(outputs, W_y.shape[0])
        e = (p_y - I)

        # backward pass, with gradient computation
        e_h_next = np.zeros_like(h[:, 0])
        for t in reversed(xrange(nr_steps)):

            # Backprop output layer
            e_h = np.dot(W_y.T, e[:, t]) + e_h_next
            # backprop through nonlinearity.
            e_raw = self.derivate_activation(
                h[:, t+1], self.activation_function) * e_h
            # Backprop through the RNN linear layer
            e_h_next = np.dot(W_h.T, e_raw)

            # Weight gradients
            nabla_W_y += np.outer(e[:, t], h[:, t+1])
            nabla_W_h += np.outer(e_raw, h[:, t])
            nabla_W_x += np.outer(e_raw, z[:, t])
            nabla_W_e[:, x[t]] += W_x.T.dot(e_raw)

        # Normalize over sentence length
        nabla_params = [nabla_W_e/nr_steps, nabla_W_x/nr_steps,
                        nabla_W_h/nr_steps, nabla_W_y/nr_steps]
        return nabla_params



