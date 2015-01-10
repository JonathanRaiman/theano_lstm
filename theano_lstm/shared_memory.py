from multiprocessing import sharedctypes
from numpy import ctypeslib
import numpy as np

def wrap_params(params):
    """
    For each parameter in a list of Theano TensorSharedVariable
    we substitute the memory with a sharedctype using the
    multiprocessing library.
    
    The wrapped memory can then be used by other child processes
    thereby synchronising different instances of a model across
    processes (e.g. for multi cpu gradient descent using single cpu
    Theano code).
    
    Inputs:
    -------
    
    params list<TensorSharedVariable /> : the list of shared Theano
                                          variables
    
    Outputs:
    --------
    
    wrapped_instances list<multiprocessing.sharedctypes> : list of
        sharedctypes (shared memory arrays) that point to the memory
        used by the current process's Theano variable.

    Usage:
    ------
    
        # define some theano model:
        mymodel = MyModel(20, 50, etc...)
    
        # wrap the memory of the Theano variables:
        shared_ctypes = wrap_params(mymodel.params)
    
    Then you can use this memory in child processes
    (See usage of `borrow_memory`)
    
    """
    wrapped_instances = []
    for param in params:
        original = param.get_value(True,True) 
        size = original.size
        shape = original.shape
        original.shape = size
        ctypes = sharedctypes.RawArray('f' if original.dtype == np.float32 else 'd', original)
        wrapped = np.frombuffer(ctypes, dtype=original.dtype, count=size)
        wrapped.shape = shape
        param.set_value(wrapped, borrow=True)
        wrapped_instances.append(ctypes)
        
    return wrapped_instances

def borrow_memory(param, memory):
    """
    Spawn different processes with the shared memory
    of your theano model's variables.

    Inputs:
    -------

    param          TensorSharedVariable : the Theano shared variable where
                                          shared memory should be used instead.
    memory multiprocessing.sharedctypes : the memory shared across processes (e.g.
                                          from `wrap_params`)

    Outputs:
    --------

    None

    Usage
    -----

    For each process in the target function run the theano_borrow_memory
    method on the parameters you want to have share memory across processes.

    In this example we have a model called "mymodel" with parameters stored in
    a list called "params". We loop through each theano shared variable and
    call `theano_borrow_memory` on it to share memory across processes.

        def spawn_model(path, wrapped_params):
            # prevent recompilation and arbitrary locks
            theano.config.reoptimize_unpickled_function = False
            theano.gof.compilelock.set_lock_status(False)

            # load your model from its pickled instance (from path)
            mymodel = MyModel.load(path)
            
            # for each parameter in your model
            # apply the borrow memory strategy to replace
            # the internal parameter's memory with the
            # across-process memory
            for param, memory in zip(mymodel.params, wrapped_params):
                borrow_memory(param, memory)
            
            # acquire your dataset (either through some smart shared memory
            # or by reloading it for each process)
            dataset, dataset_labels = acquire_dataset()
            
            # then run your model forward in this process
            epochs = 20
            for epoch in range(epochs):
                model.update_fun(dataset, dataset_labels)

    See `borrow_all_memories` for list usage.

    """

    param_value = ctypeslib.as_array(memory)
    param_value.shape = param.get_value(True,True).shape
    param.set_value(param_value, borrow=True)
    

def borrow_all_memories(params, memory_handlers):
    """
    Run theano_borrow_memory on a list of params and shared memory
    sharedctypes.

    Inputs:
    -------

    param  list<TensorSharedVariable>         : list of Theano shared variable where
                                                shared memory should be used instead.
    memory list<multiprocessing.sharedctypes> : list of memory shared across processes (e.g.
                                                from `wrap_params`)

    Outputs:
    --------

    None

    Usage:
    ------

    Same as `borrow_memory` but for lists of shared memories and
    theano variables. See `borrow_memory`

    """
    for param, memory_handler in zip(params, memory_handlers):
        borrow_memory(param, memory_handler)
