Small Theano LSTM recurrent network module
------------------------------------------

@author: Jonathan Raiman
@date: December 10th 2014

Implements most of the great things that came out
in 2014 concerning recurrent neural networks, and
some good optimizers for these types of networks.

### Key Features

This module contains several Layer types that are useful
for prediction and modeling from sequences:

* A non-recurrent **Layer**, with a connection matrix W, and bias b
* A recurrent **RNN Layer** that takes as input its previous hidden activation and has an initial hidden activation
* A recurrent **LSTM Layer** that takes as input its previous hidden activation and memory cell values, and has initial values for both of those
* An **Embedding** layer that contains an embedding matrix and takes integers as input and returns slices from its embedding matrix (e.g. word vectors)
* A non-recurrent **GatedInput**, with a connection matrix W, and bias b, that multiplies a single scalar to each input (gating jointly multiple inputs)
* Deals with exploding and vanishing gradients with a subgradient optimizer (Adadelta) and element-wise gradient clipping (à la Alex Graves)

This module also contains the **SGD**, **AdaGrad**, and **AdaDelta** gradient descent methods that are constructed using an objective function and a set of theano variables, and returns an `updates` dictionary to pass to a theano function.


### Quick Tutorial

See [a short tutorial for sequence forecasting here](http://nbviewer.ipython.org/github/JonathanRaiman/theano_lstm/blob/master/Tutorial.ipynb).
Or read on for some usage examples.

### Usage

Here is an example of usage with stacked LSTM units, using
Adadelta to optimize, and using a scan operation from Theano (a symbolic loop for backpropagation through time).

	dropout = 0.0

	model = StackedCells(4, layers=[20, 20], activation=T.tanh, celltype=LSTM)
	model.layers[0].in_gate2.activation = lambda x: x
	model.layers.append(Layer(20, 2, lambda x: T.nnet.softmax(x)[0]))

	# in this example dynamics is a random function that takes our
	# output along with the current state and produces an observation
	# for t + 1

	def step(x, *prev_hiddens):
	    new_states = stacked_rnn.forward(x, prev_hiddens, dropout)
	    return [dynamics(x, new_states[-1])] + new_states[:-1]

	initial_obs = T.vector()
	timesteps = T.iscalar()

	result, updates = theano.scan(step,
                              n_steps=timesteps,
                              outputs_info=[dict(initial=initial_obs, taps=[-1])] + [dict(initial=layer.initial_hidden_state, taps=[-1]) for layer in model.layers if hasattr(layer, 'initial_hidden_state')])

	target = T.vector()

	cost = (result[0][:,[0,2]] - target[[0,2]]).norm(L=2) / timesteps

	updates, gsums, xsums, lr, max_norm = \
		create_optimization_updates(cost, model.params, method='adadelta')

	update_fun = theano.function([initial_obs, target, timesteps], cost, updates = updates, allow_input_downcast=True)
	predict_fun = theano.function([initial_obs, timesteps], result[0], allow_input_downcast=True)

	for example, label in training_set:
		c = update_fun(example, label, 10)

### Minibatch usage

Suppose you now have many sequences (of equal length -- we'll generalize this later). Then training can be done in batches:

	model = StackedCells(4, layers=[20, 20], activation=T.tanh, celltype=LSTM)
	model.layers[0].in_gate2.activation = lambda x: x
	model.layers.append(Layer(20, 2, lambda x: T.nnet.softmax(x)[0]))

	# in this example dynamics is a function that simulates the behavior of a double
    # pendulum and takes our current state and produces an observation
	# for t + 1
    def dynamics(x, u):
        dydx = T.alloc(0.0, 4)
        dydx = T.set_subtensor(dydx[0], x[1])
        del_ = x[2]-x[0]
        den1 = (M1+M2)*L1 - M2*L1*T.cos(del_)*T.cos(del_)
        dydx = T.set_subtensor(dydx[1],\n",
            (  M2*L1      *  x[1] * x[1] * T.sin(del_) * T.cos(del_)
               + M2*G       *  T.sin(x[2]) * T.cos(del_) +
                 M2*L2      *  x[3] * x[3] * T.sin(del_)
               - (M1+M2)*G  *  T.sin(x[0]))/den1 )
        dydx = T.set_subtensor(dydx[2], x[3])

        den2 = (L2/L1)*den1
        dydx = T.set_subtensor(dydx[3], (-M2*L2  *   x[3]*x[3]*T.sin(del_) * T.cos(del_)
                   + (M1+M2)*G   *   T.sin(x[0])*T.cos(del_)
                   - (M1+M2)*L1  *   x[1]*x[1]*T.sin(del_)
                   - (M1+M2)*G   *   T.sin(x[2]))/den2  + u )
        return x + dydx * dt

	def step(x, *prev_hiddens):
	    new_states = stacked_rnn.forward(x, prev_hiddens, dropout)
	    return [dynamics(x, new_states[-1])] + new_states[:-1]

	# switch to a matrix of observations:
	initial_obs = T.imatrix()
	timesteps = T.iscalar()

	result, updates = theano.scan(step,
                              n_steps=timesteps,
                              outputs_info=[dict(initial=initial_obs, taps=[-1])] + [dict(initial=layer.initial_hidden_state, taps=[-1]) for layer in model.layers if hasattr(layer, 'initial_hidden_state')])

	target = T.ivector()

	cost = (result[0][:,:,[0,2]] - target[:,[0,2]]).norm(L=2) / timesteps

	updates, gsums, xsums, lr, max_norm = \
		create_optimization_updates(cost, model.params, method='adadelta')

	update_fun = theano.function([initial_obs, target, timesteps], cost, updates = updates, allow_input_downcast=True)
	predict_fun = theano.function([initial_obs, timesteps], result[0], allow_input_downcast=True)

	for minibatch, labels in minibatches:
		c = update_fun(minibatch, label, 10)

### Minibatch usage with different sizes

Generalization can be made to different sequence length if we accept the minor cost of forward-propagating parts of our graph we don't care about. To do this we make all sequences the same length by padding the end of the shorter ones with some symbol. Then use a binary matrix of the same size than all your minibatch sequences. The matrix has a 1 in areas when the error should be calculated, and zero otherwise. Elementwise mutliply this mask with your output, and then apply your objective function to this masked output. The error will be obtained everywhere, but will be zero in areas that were masked, yielding the correct error function.
While there is some waste computation, the parallelization can offset this cost and make the overall computation faster.

#### MaskedLoss usage

To use different length sequences, consider the following approach:

* you have sequences *y_1, y_2, ..., y_n*, and labels *l_1, l_2, ..., l_n*.
* pad all the sequences to the longest sequence *y_k*, and form a matrix **Y** of all padded sequences
* similarly form the labels at each timestep for each padded sequence (with zeros, or some other symbol for labels in padded areas)
* then record the length of the true labels (codelengths) needed before padding *c_1, c_2, ..., c_n*, and the length of the sequences before padding *l_1, l_2, ..., l_n*
* pass the lengths, targets, and predictions to the masked loss as follows:

		predictions, updates = theano.scan(prediction_step, etc...)

		error = masked_loss(
	            predictions,
	            padded_labels,
	            codelengths,
	            label_starts).mean()

Visually this goes something like this, for the case with three inputs, three outputs, but a single label for
the final output:

inputs  [ x_1 x_2 x_3 ]

outputs [ p_1 p_2 p_3 ]

labels  [ ... ... l_1 ]

then we would have a matrix *x* with *x_1, x_2, x_3*, and `predictions` in the code above would contain *p_1, p_2, p_3*.
We would then pass to `masked_loss` the codelength [ 1 ], since there is only "l_1" to predict, and the `label_starts` [ 2 ],
indicating that errors should be computed at the third prediction (with zero index).

#### Dropout Usage in Theano Scan

To get dropout to work and be dynamically modifyiable without recompiling let's consider the following usage example.

First we define a variable with the likelihood that a neuron will be dropped (randomly set to 0):

	dropout = theano.shared(np.float64(0.3).astype(theano.config.floatX))
	deterministic = False # for now

Create some model:

	model = theano_lstm.StackedCells(50, layers=[100], celltype=theano_lstm.LSTM, activation=T.tanh)

Now we want to introduce dropout noise between the input and the LSTM. To use Dropout outside of a Theano `scan` loop you could simply multiply elementwise by a binomial random variable ([see examples here](https://gist.github.com/SnippyHolloW/8a0f820261926e2f41cc)), but if you plan on using recurrent networks with a Theano `scan` you need to call your random numbers outside of the loop.

In order to keep track of these dropout activations we'll generate *masks*. *Masks* are a list with all the realizations of binomials. We generate this list with `MultiDropout`, a special function in the `theano_lstm` module that takes different hidden layer sizes and returns a list of matrices with binomial random variable realizations inside:

	if dropout.get_value() > 0:
        if deterministic:
            # just multiply by the likelihood of being kept:
            masks = [np.float32(1.) - self.dropout for i in range(2)]
        else:
            shapes = [50, 100]
            masks = theano_lstm.MultiDropout( [(x.shape[0], shape) for shape in shapes] if x.ndim > 1 else shapes,
                                                            self.dropout)
    else:
        masks = []

Now our loop forward function is as follows:

	def step(obs, hidden_state, *masks):
        new_state = model.forward(obs, [hidden_state], list(masks))
        return new_state[1]

We pass it to Theano's scan:

    result, _ = theano.scan(step,
    	sequences     = seq,
    	non_sequences = masks,
    	outputs_info  = [dict(initial=model.layers[0].initial_hidden_state, taps=[-1])]
    	)

And We're done.

**Note:** To not use *Masks* pass an empty list `[]` instead.
