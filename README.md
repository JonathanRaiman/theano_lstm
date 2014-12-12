Small Theano LSTM recurrent network module
------------------------------------------

@author: Jonathan Raiman
@date: December 10th 2014

Implements most of the great things that came out
in 2014 concerning recurrent neural networks, and
some good optimizers for these types of networks.

**Note**: Dropout causes gradient issues with theano
if placed in scan, so it should be set to 0 for now,
and will be fixed in the future.

### Key Features

This module contains several Layer types that are useful
for prediction and modeling from sequences:

* A non-recurrent **Layer**, with a connection matrix W, and bias b
* A recurrent **RNN Layer** that takes as input its previous hidden activation and has an initial hidden activation
* A recurrent **LSTM Layer** that takes as input its previous hidden activation and memory cell values, and has initial values for both of those
* An **Embedding** layer that contains an embedding matrix and takes integers as input and returns slices from its embedding matrix (e.g. word vectors)

This module also contains the **SGD**, **AdaGrad**, and **AdaDelta** gradient descent methods that are constructed using an objective function and a set of theano variables, and returns an `updates` dictionary to pass to a theano function (see below).


### Usage

Here is an example of usage with stacked LSTM units, using
Adadelta to optimize, and using a scan op.


	# bug for now forces us to use 0.0 with scan,
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