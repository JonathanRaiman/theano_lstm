"""
Small Theano LSTM recurrent network module.

@author: Jonathan Raiman
@date: December 10th 2014

Implements most of the great things that came out
in 2014 concerning recurrent neural networks, and
some good optimizers for these types of networks.

Note: Dropout causes gradient issues with theano if placed
      in scan, so it should be set to 0 for now, and will be fixed
      in the future.

"""

import theano, theano.tensor as T
import numpy as np
from collections import OrderedDict

srng = theano.tensor.shared_randomstreams.RandomStreams(1234)

def create_shared(out_size, in_size = None):
    """
    Creates a shared matrix or vector
    using the given in_size and out_size.

    Inputs
    ------

    out_size int            : outer dimension of the
                              vector or matrix
    in_size  int (optional) : for a matrix, the inner
                              dimension.

    Outputs
    -------

    theano shared : the shared matrix, with random numbers in it

    """

    if in_size is None:
        return theano.shared((np.random.standard_normal([out_size])* 1./out_size).astype(theano.config.floatX))
    else:
        return theano.shared((np.random.standard_normal([out_size, in_size])* 1./out_size).astype(theano.config.floatX))
    
def Dropout(x, prob):
    """
    Perform dropout (binomial noise) on x.

    The probability of a value in x going to zero is prob.

    Inputs
    ------

    x    theano variable : the variable to add noise to
    prob float, variable : probability of dropping an element.


    Outputs
    -------

    y    theano variable : x with the noise multiplied.

    """
    
    mask = srng.binomial(n=1, p=1-prob, size=x.shape)
    y = x * T.cast(mask, theano.config.floatX)
    return y

class Layer(object):
    """
    Base object for neural network layers.

    A layer has an input set of neurons, and
    a hidden activation. The activation, f, is a
    function applied to the affine transformation
    of x by the connection matrix W, and the bias
    vector b.

    > y = f ( W * x + b )

    """
        
    def __init__(self, input_size, hidden_size, activation):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.activation  = activation
        self.is_recursive = False
        self.create_variables()
        
    def create_variables(self):
        """
        Create the connection matrix and the bias vector
        """
        self.params = []
        self.linear_matrix        = create_shared(self.hidden_size, self.input_size)
        self.bias_matrix          = create_shared(self.hidden_size)
        self.params.append(self.linear_matrix)
        self.params.append(self.bias_matrix)
        
    def activate(self, x):
        """
        The hidden activation of the network
        """
        return self.activation(
            T.dot(self.linear_matrix, x) + self.bias_matrix )
    
class RNN(Layer):
    """
    Special recurrent layer than takes as input
    a hidden activation, h, from the past and
    an observation x.

    > y = f ( W * [x, h] + b )

    Note: x and h are concatenated in the activation.

    """
    def __init__(self, *args):
        super().__init__(*args)
        self.is_recursive = True
        
    def create_variables(self):
        """
        Create the connection matrix and the bias vector,
        and the base hidden activation.

        """
        self.params = []
        self.linear_matrix        = create_shared(self.hidden_size, self.input_size+ self.hidden_size)
        self.bias_matrix          = create_shared(self.hidden_size)
        self.params.append(self.linear_matrix)
        self.params.append(self.bias_matrix)
        self.initial_hidden_state = create_shared(self.hidden_size)
        self.params.append(self.initial_hidden_state)
        
    def activate(self, x, h):
        """
        The hidden activation of the network
        """
        return self.activation(
            T.dot(
                self.linear_matrix,
                T.concatenate([x, h])
            ) + self.bias_matrix )
    
class LSTM(RNN):
    """
    The structure of the LSTM allows it to learn on problems with
    long term dependencies relatively easily. The "long term"
    memory is stored in a vector of memory cells c.
    Although many LSTM architectures differ in their connectivity
    structure and activation functions, all LSTM architectures have
    memory cells that are suitable for storing information for long
    periods of time. Here we implement the LSTM from Graves et al.
    (2013).
    """
        
    def create_variables(self):
        """
        Create the different LSTM gates and
        their variables, along with the initial
        hidden state for the memory cells and
        the initial hidden activation.

        """
        # input gate for cells
        self.in_gate     = Layer(self.input_size + self.hidden_size, self.hidden_size, T.nnet.sigmoid)
        # forget gate for cells
        self.forget_gate = Layer(self.input_size + self.hidden_size, self.hidden_size, T.nnet.sigmoid)
        # input modulation for cells
        self.in_gate2    = Layer(self.input_size + self.hidden_size, self.hidden_size, self.activation)
        # output modulation
        self.out_gate    = Layer(self.input_size + self.hidden_size, self.hidden_size, T.nnet.sigmoid)
        
        # keep these layers organized
        self.internal_layers = [self.in_gate, self.forget_gate, self.in_gate2, self.out_gate]
        
        # store the memory cells in first n spots, and store the current
        # output in the next n spots:
        self.initial_hidden_state = create_shared(self.hidden_size * 2)
        
    @property
    def params(self):
        """
        Parameters given by the 4 gates and the
        initial hidden activation of this LSTM cell
        layer.

        """
        return [self.initial_hidden_state] + [param for layer in self.internal_layers for param in layer.params]
        
    def activate(self, x, h):
        """
        The hidden activation, h, of the network, along
        with the new values for the memory cells, c,
        Both are concatenated as follows:

        >      y = f( x, past )

        Or more visibly, with past = [prev_c, prev_h]

        > [c, h] = f( x, [prev_c, prev_h] )

        """
        #previous memory cell values
        prev_c = h[:self.hidden_size]
        
        #previous activations of the hidden layer
        prev_h = h[self.hidden_size:]
        
        # input and previous hidden constitute the actual
        # input to the LSTM:
        obs = T.concatenate([x, prev_h])
        
        # how much to add to the memory cells
        in_gate = self.in_gate.activate(obs)
        
        # how much to forget the current contents of the memory
        forget_gate = self.forget_gate.activate(obs)
        
        # modulate the input for the memory cells
        in_gate2 = self.in_gate2.activate(obs)
        
        # new memory cells
        next_c = forget_gate * prev_c + in_gate2 * in_gate
        
        # modulate the memory cells to create the new output
        out_gate = self.out_gate.activate(obs)
        
        # new hidden output
        next_h = out_gate * T.tanh(next_c)
        
        return T.concatenate([next_c, next_h])

class StackedCells(object):
    """
    Sequentially connect several recurrent layers.

    celltypes can be RNN or LSTM.

    """
    def __init__(self, input_size, celltype=RNN, layers = [], activation = lambda x:x):
        self.input_size = input_size
        self.create_layers(layers, activation, celltype)
        
    def create_layers(self, layer_sizes, activation_type, celltype):
        self.layers = []
        prev_size   = self.input_size
        for k, layer_size in enumerate(layer_sizes):
            layer = celltype(prev_size, layer_size, activation_type)
            self.layers.append(layer)
            prev_size = layer_size
    
    @property
    def params(self):
        return [param for layer in self.layers for param in layer.params] 
            
    def forward(self, x, prev_hiddens = None, dropout = 0.0):
        """
        Return new hidden activations for all stacked RNNs
        """
        
        if prev_hiddens is None:
            prev_hiddens = [layer.initial_hidden_state for layer in self.layers if hasattr(layer, 'initial_hidden_state')]
        
        out = []
        layer_input = x
        for k, layer in enumerate(self.layers):
            if dropout > 0 and k > 0:
                layer_input = Dropout(layer_input, dropout)
            if layer.is_recursive:
                layer_input = layer.activate(layer_input, prev_hiddens[k])
            else:
                layer_input = layer.activate(layer_input)
            out.append(layer_input)
            # deliberate choice to change the upward structure here
            # in an RNN, there is only one kind of hidden values
            if type(layer) is LSTM:
                # in this case the hidden activation has memory cells
                # that are not shared upwards
                # along with hidden activations that can be sent
                # updwards
                layer_input = layer_input[layer.hidden_size:]
        return out

def create_optimization_updates(cost, params, max_norm = 5.0, lr = 0.01, eps= 1e-6, rho=0.95, method = "adadelta"):
    """
    Get the updates for a gradient descent optimizer using
    SGD, AdaDelta, or AdaGrad.

    Returns the shared variables for the gradient caches,
    and the updates dictionary for compilation by a
    theano function.

    Inputs
    ------

    cost     theano variable : what to minimize
    params   list            : list of theano variables
                               with respect to which
                               the gradient is taken.
    max_norm float           : cap on excess gradients
    lr       float           : base learning rate for
                               adagrad and SGD
    eps      float           : numerical stability value
                               to not divide by zero
                               sometimes
    rho      float           : adadelta hyperparameter.
    method   str             : 'adagrad', 'adadelta', or 'sgd'.


    Outputs:
    --------

    updates  OrderedDict   : the updates to pass to a
                             theano function
    gsums    list          : gradient caches for Adagrad
                             and Adadelta
    xsums    list          : gradient caches for AdaDelta only
    lr       theano shared : learning rate
    max_norm theano_shared : normalizing clipping value for
                             excessive gradients (exploding).

    """
    lr = theano.shared(np.float64(lr).astype(theano.config.floatX))
    eps = np.float64(eps).astype(theano.config.floatX)
    rho = np.float64(rho).astype(theano.config.floatX)
    max_norm = theano.shared(np.float64(max_norm).astype(theano.config.floatX))

    gsums   = [theano.shared(np.zeros_like(param.get_value(borrow=True))) if (method == 'adadelta' or method == 'adagrad') else None for param in params]
    xsums   = [theano.shared(np.zeros_like(param.get_value(borrow=True))) if method == 'adadelta' else None for param in params]

    gparams = T.grad(cost, params)
    updates = OrderedDict()

    for gparam, param, gsum, xsum in zip(gparams, params, gsums, xsums):
        # clip gradients if they get too big
        grad_norm = gparam.norm(L=2)
        gparam = (T.minimum(max_norm, grad_norm)/ grad_norm) * gparam
        
        if method == 'adadelta':
            updates[gsum] = T.cast(rho * gsum + (1. - rho) * (gparam **2), theano.config.floatX)
            dparam = -T.sqrt((xsum + eps) / (updates[gsum] + eps)) * gparam
            updates[xsum] = T.cast(rho * xsum + (1. - rho) * (dparam **2), theano.config.floatX)
            updates[param] = T.cast(param + dparam, theano.config.floatX)
        elif method == 'adagrad':
            updates[gsum] =  T.cast(gsum + (gparam ** 2), theano.config.floatX)
            updates[param] =  T.cast(param - lr * (gparam / (T.sqrt(updates[gsum] + eps))), theano.config.floatX)
        else:
            updates[param] = param - gparam * lr

    return updates, gsums, xsums, lr, max_norm