import numpy as np
import cPickle


def index2onehot(index, N):
    """
    Transforms index to one-hot representation, for example

    Input: e.g. index = [1, 2, 0], N = 4
    Output:     [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]
    """
    L = index.shape[0]
    onehot = np.zeros((N, L))
    for l in np.arange(L):
        onehot[index[l], l] = 1
    return onehot


def glorot_weight_init(num_inputs, num_outputs, activation_function,
                       random_seed):
    """
    Layer weight initialization after Xavier Glorot et. al
    """

    weight = random_seed.uniform(
        low=-np.sqrt(6. / (num_inputs + num_outputs)),
        high=np.sqrt(6. / (num_inputs + num_outputs)),
        size=(num_outputs, num_inputs)
    )
    if activation_function == 'sigmoid':
        weight *= 4
    elif activation_function == 'softmax':
        weight *= 4

    return weight


class MLP():
    """
    Basic MLP class mathods
    """

    def __init__(self, geometry, activation_function, rng=None,
                 model_file=None):
        """
        Input: geometry  tuple with sizes of layer

        Input: activation_function  list of strings indicating the type of
                                    activation function. Supported 'sigmoid',
                                    'softmax'

        Input: rng       string indicating random seed
        """
        pass

    def forward(self, x, all_outputs=False):
        """
        Forward pass

        all_outputs = True  return intermediate activations
        """
        raise Exception("Implement forward in the child class")

    def gradients(self, x, y):
        """
        Computes the gradients of the network with respect to cross entropy
        error cost
        """
        raise Exception("Implement gradients in the child class")

    def init_weights(self, geometry, activation_functions, random_seed=None):

        # Initialize random seed if not given
        if not random_seed:
            random_seed = np.random.RandomState(1234)

        parameters = []
        for n in range(self.n_layers):

            # Weights
            weight = glorot_weight_init(
                geometry[n],
                geometry[n+1],
                activation_functions[n],
                random_seed
            )

            # Bias
            bias = np.zeros((geometry[n+1], 1))

            # Append parameters
            parameters.append(weight)
            parameters.append(bias)

        return parameters, activation_functions

    def sanity_checks(self, geometry, activation_function, model_file):

        # CHECK GENRERAL CONFIGURATION
        if model_file and (geometry or activation_function):
            raise ValueError(
                "If you load a model geometry and activation_function"
                "should be None"
            )

        # CHECK ACTIVATIONS
        if activation_function:
            # Supported activation_function
            supported_acts = ['sigmoid', 'softmax']
            if geometry and (len(activation_function) != len(geometry)-1):
                raise ValueError(
                    "The number of layers and activation_function does not"
                    " match"
                )
            elif any(
                [act not in supported_acts for act in activation_function]
            ):
                raise ValueError(
                    "Only these activation_function supported %s" %
                    (" ".join(supported_acts))
                )
            # All internal layers must be a sigmoid
            for internal_act in activation_function[:-1]:
                if internal_act != 'sigmoid':
                    raise ValueError("Intermediate layers must be sigmoid")

    def save(self, model_path):
        """
        Save model
        """
        par = self.parameters + self.activation_function
        with open(model_path, 'wb') as fid:
            cPickle.dump(par, fid, cPickle.HIGHEST_PROTOCOL)

    def load(self, model_path):
        """
        Load model
        """
        with open(model_path) as fid:
            par = cPickle.load(fid, cPickle.HIGHEST_PROTOCOL)
            parameters = par[:len(par)//2]
            activation_function = par[len(par)//2:]
        return parameters, activation_function

    def plot_weights(self, show=True, aspect='auto'):
        """
        Plots the weights of the newtwork
        """
        import matplotlib.pyplot as plt
        plt.figure()
        for n in range(self.n_layers):
            # Get weights
            W = self.parameters[2*n]
            b = self.parameters[2*n+1]

            plt.subplot(2, self.n_layers, n+1)
            plt.imshow(W, aspect=aspect, interpolation='nearest')
            plt.title('Layer %d Weight' % n)
            plt.colorbar()
            plt.subplot(2, self.n_layers, self.n_layers+(n+1))
            plt.plot(b)
            plt.title('Layer %d Bias' % n)
            plt.colorbar()
        if show:
            plt.show()
