"""
Basic MLP class methods for parameters initialization, saving, loading
plotting
"""
import os
import cPickle
import yaml
import numpy as np
from lxmls.deep_learning.utils import Model, glorot_weight_init


def load_parameters(parameter_file):
    """
    Load model
    """
    with open(parameter_file, 'rb') as fid:
        parameters = cPickle.load(fid)
    return parameters


def load_config(config_path):
    with open(config_path, 'r') as fid:
        config = yaml.load(fid)
    return config


def save_config(config_path, config):
    with open(config_path, 'w') as fid:
        yaml.dump(config, fid, default_flow_style=False)


def initialize_mlp_parameters(geometry, activation_functions,
                              loaded_parameters=None, random_seed=None):
    """
    Initialize parameters from geometry or existing weights
    """

    # Initialize random seed if not given
    if random_seed is None:
        random_seed = np.random.RandomState(1234)

    if loaded_parameters is not None:
        assert len(loaded_parameters) == len(activation_functions), \
            "New geometry not matching model saved"

    parameters = []
    num_layers = len(activation_functions)
    for n in range(num_layers):

        # Weights
        if loaded_parameters is not None:
            weight, bias = loaded_parameters[n]
            assert weight.shape == (geometry[n+1], geometry[n]), \
                "New geometry does not match for weigths in layer %d" % n
            assert bias.shape == (1, geometry[n+1]), \
                "New geometry does not match for bias in layer %d" % n

        else:
            weight = glorot_weight_init(
                (geometry[n], geometry[n+1]),
                activation_functions[n],
                random_seed
            )

            # Bias
            bias = np.zeros((1, geometry[n+1]))

        # Append parameters
        parameters.append([weight, bias])

    return parameters


def mlp_parameter_handlers(layer_index=None, is_bias=None, row=None,
                           column=None):

    def get_parameter(parameters):
        if is_bias:
            # bias
            return parameters[layer_index][is_bias][row]
        else:
            # weight
            return parameters[layer_index][is_bias][row, column]

    def set_parameter(parameters, parameter_value):
        if is_bias:
            # bias
            parameters[layer_index][is_bias][row] = parameter_value
        else:
            # weight
            parameters[layer_index][is_bias][row, column] = parameter_value
        return parameters

    return get_parameter, set_parameter


class MLP(Model):
    def __init__(self, **config):

        # CHECK THE PARAMETERS ARE VALID
        self.sanity_checks(config)

        # OPTIONAL MODEL LOADING
        model_folder = config.get('model_folder', None)
        if model_folder is not None:
            saved_config, loaded_parameters = self.load(model_folder)
            # Note that if a config is given this is used instead of the saved
            # one (must be consistent)
            if config is None:
                config = saved_config
        else:
            loaded_parameters = None

        # MEMBER VARIABLES
        self.num_layers = len(config['activation_functions'])
        self.config = config
        self.parameters = initialize_mlp_parameters(
            config['geometry'],
            config['activation_functions'],
            loaded_parameters
        )

    def sanity_checks(self, config):

        model_folder = config.get('model_folder', None)

        assert bool(config is None) or bool(model_folder is None), \
            "Need to specify config, model_folder or both"

        if config is not None:

            geometry = config['geometry']
            activation_functions = config['activation_functions']

            assert len(activation_functions) == len(geometry) - 1, \
                "geometry and activation_functions sizs do not match"

            assert all(
                afun == 'sigmoid'
                for afun in activation_functions[:-1]
            ), "Hidden layer activations must be sigmoid"

            assert activation_functions[-1] in ['sigmoid', 'softmax'], \
                "Output layer activations must be sigmoid or softmax"

        if model_folder is not None:
            model_file = "%s/config.yml" % model_folder
            assert os.path.isfile(model_file), "Need to provide %s" % model_file

    def load(self, model_folder):
        """
        Load model
        """

        # Configuration un yaml format
        config_file = "%s/config.yml" % model_folder
        config = load_config(config_file)

        # Computation graph parameters as pickle file
        parameter_file = "%s/parameters.pkl" % model_folder
        loaded_parameters = load_parameters(parameter_file)

        return config, loaded_parameters

    def save(self, model_folder):
        """
        Save model
        """

        # Create folder if it does not exist
        if not os.path.isdir(model_folder):
            os.mkdir(model_folder)

        # Configuration un yaml format
        config_file = "%s/config.yml" % model_folder
        save_config(config_file, self.config)

        # Computation graph parameters as pickle file
        parameter_file = "%s/parameters.pkl" % model_folder
        with open(parameter_file, 'wb') as fid:
            cPickle.dump(self.parameters, fid, cPickle.HIGHEST_PROTOCOL)

    def plot_weights(self, show=True, aspect='auto'):
        """
        Plots the weights of the newtwork

        Use show = False to plot various models one after the other
        """
        import matplotlib.pyplot as plt
        plt.figure()
        for n in range(self.n_layers):

            # Get weights and bias
            weight, bias = self.parameters[n]

            # Plot them
            plt.subplot(2, self.n_layers, n+1)
            plt.imshow(weight, aspect=aspect, interpolation='nearest')
            plt.title('Layer %d Weight' % n)
            plt.colorbar()
            plt.subplot(2, self.n_layers, self.n_layers+(n+1))
            plt.plot(bias)
            plt.title('Layer %d Bias' % n)
            plt.colorbar()

        if show:
            plt.show()
