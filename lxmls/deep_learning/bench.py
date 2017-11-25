import numpy as np


def categorical_scores(prob_y, y_ref):

    assert prob_y.shape[0] == y_ref.shape[0], \
        "Class probabilities and reference class sizes do not match"

    # Average probability set
    pred = prob_y[np.arange(y_ref.shape[0]), y_ref]
    log_prob = np.mean(np.log(pred))

    # Accuracy in set
    hat_y = np.argmax(prob_y, 1)
    accuracy = np.mean(hat_y == y_ref)

    return accuracy, log_prob


class Data(object):
    """
    Template
    """
    def __init__(self, **kwargs):

        # Create config from input arguments
        config = kwargs

        # Data-sets
        self.datasets = {
            'train': {
                'input': config['corpus'].train_X,
                'output': config['corpus'].train_y[:, 0]
            },
            #  'dev': (config['corpus'].dev_X, config['corpus'].dev_y[:, 0]),
            'test': {
                'input': config['corpus'].test_X,
                'output': config['corpus'].test_y[:, 0]
            }
        }
        # Config
        self.config = config
        # Number of samples
        self.nr_samples = {
           sset: content['output'].shape[0]
           for sset, content in self.datasets.items()
        }

    def size(self, set_name):
        return self.nr_samples[set_name]

    def batches(self, set_name, batch_size=None):

        dset = self.datasets[set_name]
        nr_examples = self.nr_samples[set_name]
        if batch_size is None:
            nr_batch = 1
            batch_size = nr_examples
        else:
            nr_batch = int(np.ceil(nr_examples*1./batch_size))

        data = []
        for batch_n in range(nr_batch):
            # Colect data for this batch
            data_batch = {}
            for side in ['input', 'output']:
                data_batch[side] = dset[side][
                   batch_n * batch_size:(batch_n + 1) * batch_size
                ]
            data.append(data_batch)

        return DataIterator(data, nr_samples=self.nr_samples[set_name])


class DataIterator(object):
    """
    Basic data iterator
    """

    def __init__(self, data, nr_samples):
        self.data = data
        self.nr_samples = nr_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class Model(object):

    def __init__(self, config=None, model_folder=None):
        assert bool(model_folder) or bool(config), \
            "Provide either model_folder, config or both"
        if model_folder is not None:
            self.load(model_folder)
        else:
            self.config = config
        self.initialized = False

    def initialize_features(self, *args):
        self.initialized = True
        raise NotImplementedError(
            "Need to implement initialize_features method"
        )

    def get_features(self, input=None, output=None):
        """
        Default feature extraction is do nothing
        """
        return {'input': input, 'output': output}

    def predict(self, *args):
        raise NotImplementedError("Need to implement predict method")

    def update(self, *args):
        # This needs to return at least {'cost' : 0}
        raise NotImplementedError("Need to implement update method")
        return {'cost': None}

    def set(self, **kwargs):
        raise NotImplementedError("Need to implement set method")

    def get(self, name):
        raise NotImplementedError("Need to implement get method")

    def save(self):
        raise NotImplementedError("Need to implement save method")

    def load(self, model_folder):
        raise NotImplementedError("Need to implement load method")
