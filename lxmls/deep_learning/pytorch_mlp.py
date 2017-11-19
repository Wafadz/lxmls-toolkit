from __future__ import division
import torch
from torch.autograd import Variable


class PytorchMLP(NumpyMLP):
    """
    MLP VERSION USING THEANO
    """

    def __init__(self, geometry, actvfunc, rng=None, model_file=None):
        """
        Input: geometry  tuple with sizes of layer

        Input: actvfunc  list of strings indicating the type of activation
                         function. Supported 'sigmoid', 'softmax'

        Input: rng       string inidcating random seed
        """

        # Generate random seed if not provided
        if rng is None:
            rng = np.random.RandomState(1234)

        # This will call NumpyMLP.__init__.py intializing
        # Defining: self.n_layers self.params self.actvfunc
        NumpyMLP.__init__(self, geometry, actvfunc, rng=rng, model_file=model_file)

        # The parameters in the Theano MLP are stored as shared, borrowed
        # variables. This data will be moved to the GPU when used
        # use self.params.get_value() and self.params.set_value() to acces or
        # modify the data in the shared variables
        self.shared_params()

        self.cost = torch.nn.MSELoss(size_average=False)

    def shared_params(self):

        params = [None] * (2*self.n_layers)
        for n in range(self.n_layers):
            # Get Numpy weigths and bias (always in even and odd positions)
            W = self.params[2*n]
            b = self.params[2*n+1]

            # IMPORTANT: Ensure the types in the variables and theano operations
            # match. This is ofte a source of errors
            W = torch.from_numpy(W).float()
            b = torch.from_numpy(b).float()

            # Store weight and bias, now as theano shared variables
            params[2*n] = W
            params[2*n+1] = b

        # Overwrite our params
        self.params = params

    def forward(self, x, all_outputs=False):
        """
        Symbolic forward pass

        all_outputs = True  return symbolic input and intermediate activations
        """

        # Ensure the type matches torch type
        x = Variable(torch.from_numpy(x).float(), requires_grad=False)

        # This will store activations at each layer and the input. This is
        # needed to compute backpropagation
        if all_outputs:
            activations = [x]

        # Input
        tilde_z = x

        # ----------
        # Solution to Exercise 6.4
        for n in range(self.n_layers):

            # Get weigths and bias (always in even and odd positions)
            W = self.params[2*n]
            b = self.params[2*n+1]

            # Linear transformation
            z = torch.matmul(W, tilde_z) + b

            # Non-linear transformation
            if self.actvfunc[n] == "sigmoid":
                tilde_z = torch.sigmoid(z)
            elif self.actvfunc[n] == "softmax":
                import ipdb;ipdb.set_trace(context=30)
                tilde_z = torch.nn.softmax(z.T).T

            if all_outputs:
                activations.append(tilde_z)
        # End of solution to Exercise 6.4
        # ----------

        if all_outputs:
            tilde_z = activations

        return tilde_z

    def update(self, x, y):
        # TODO: Better do it operation by operation
        raise NotImplementedError()



