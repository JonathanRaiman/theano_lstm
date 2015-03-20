import numpy as np
import theano.tensor as T
from theano import gof
from theano.gof import Apply
from theano.gradient import grad_not_implemented

class MaskedLossDx(gof.Op):

    def make_node(self, softmaxes, y_idxes, y_lengths, y_startidxes, g_costs, **kwargs):
        softmaxes = T.as_tensor_variable(softmaxes)
        y_idxes = T.as_tensor_variable(y_idxes)
        y_lengths = T.as_tensor_variable(y_lengths)
        y_startidxes = T.as_tensor_variable(y_startidxes)
        g_costs = T.as_tensor_variable(g_costs)

        if (softmaxes.type.ndim != 3 or
            softmaxes.type.dtype not in T.float_dtypes):
            raise ValueError('dy must be 3-d tensor of floats', softmaxes.type)

        if (y_idxes.type.ndim != 2 or
            y_idxes.type.dtype not in T.discrete_dtypes):
            raise ValueError('y_idxes must be 2-d tensor of integers', y_idxes.type)

        if (y_lengths.type.ndim != 1 or
            y_lengths.type.dtype not in T.discrete_dtypes):
            raise ValueError('y_lengths must be 1-d tensor of integers', y_lengths.type)

        if (y_startidxes.type.ndim != 1 or
            y_startidxes.type.dtype not in T.discrete_dtypes):
            raise ValueError('y_startidxes must be 1-d tensor of integers', y_startidxes.type)

        if (g_costs.type.ndim != 1 or
            g_costs.type.dtype not in T.float_dtypes):
            raise ValueError('g_costs must be 1-d tensor of floats', g_costs.type)

        return Apply(self, [softmaxes, y_idxes, y_lengths, y_startidxes, g_costs],
                     [T.Tensor(dtype=softmaxes.dtype, broadcastable=softmaxes.type.broadcastable)()])

    def perform(self, node, input_storage, output_storage):
        softmaxes, y_idxes, y_lengths, y_startidxes, g_costs = input_storage

        dx = np.zeros_like(softmaxes)
        for i in range(y_lengths.shape[0]):
            # take the total cost to be the errors made
            #dx[i, y_startidxes[i]:y_startidxes[i]+y_lengths[i]] = softmaxes[i, y_startidxes[i]:y_startidxes[i]+y_lengths[i]] * g_costs[i]
            dx[i,
               np.arange(y_startidxes[i], y_startidxes[i] + y_lengths[i]),
               y_idxes[i, y_startidxes[i]:y_startidxes[i]+y_lengths[i]]
               ] -= 1./(softmaxes[i,
                              np.arange(y_startidxes[i], y_startidxes[i] + y_lengths[i]),
                              y_idxes[i, y_startidxes[i]:y_startidxes[i]+y_lengths[i]]] * g_costs[i])

        output_storage[0][0] = dx

    def c_code_cache_version(self):
        return (3,)

    def __init__(self, **kwargs):
        gof.Op.__init__(self, **kwargs)

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return T.hashtype(self)

    def __str__(self):
        return self.__class__.__name__

    def c_code(self, node, name, inp, out, sub):
        softmaxes, y_idxes, y_lengths, y_startidxes, g_costs = inp
        dx, = out
        out_typenum = node.inputs[0].type.dtype_specs()[2]
        return """

        if ((PyArray_TYPE(%(g_costs)s) != NPY_DOUBLE) &&
            (PyArray_TYPE(%(g_costs)s) != NPY_FLOAT))
        {
            PyErr_SetString(PyExc_TypeError,
                 "g_costs type should be float32 or float64");
            %(fail)s;
        }
        if ((PyArray_TYPE(%(softmaxes)s) != NPY_DOUBLE) &&
            (PyArray_TYPE(%(softmaxes)s) != NPY_FLOAT))
        {
            PyErr_SetString(PyExc_TypeError,
                 "softmaxes type should be float32 or float64");
            %(fail)s;
        }
        if ((PyArray_NDIM(%(g_costs)s) != 1)
            || (PyArray_NDIM(%(softmaxes)s) != 3)
            || (PyArray_NDIM(%(y_idxes)s) != 2)
            || (PyArray_NDIM(%(y_lengths)s) != 1)
            || (PyArray_NDIM(%(y_startidxes)s) != 1))
        {
            PyErr_SetString(PyExc_ValueError, "rank error");
            %(fail)s;
        }
        if (PyArray_DIMS(%(g_costs)s)[0] != PyArray_DIMS(%(softmaxes)s)[0])
        {
            PyErr_Format(PyExc_ValueError,
                         "g_costs.shape[0] (%%ld) != softmaxes.shape[0] (%%ld)",
                         (long int)PyArray_DIMS(%(g_costs)s)[0],
                         (long int)PyArray_DIMS(%(softmaxes)s)[0]);
            %(fail)s;
        }
        if (PyArray_DIMS(%(g_costs)s)[0] != PyArray_DIMS(%(y_idxes)s)[0])
        {
            PyErr_Format(PyExc_ValueError,
                         "g_costs.shape[0] (%%ld) != y_idxes.shape[0] (%%ld)",
                         (long int)PyArray_DIMS(%(g_costs)s)[0],
                         (long int)PyArray_DIMS(%(y_idxes)s)[0]);
            %(fail)s;
        }
        if ((NULL == %(dx)s)
            || (PyArray_DIMS(%(dx)s)[0] != PyArray_DIMS(%(softmaxes)s)[0])
            || (PyArray_DIMS(%(dx)s)[1] != PyArray_DIMS(%(softmaxes)s)[1])
            || (PyArray_DIMS(%(dx)s)[2] != PyArray_DIMS(%(softmaxes)s)[2]))
        {
            if (NULL != %(dx)s) Py_XDECREF(%(dx)s);
            %(dx)s = (PyArrayObject*) PyArray_Zeros(3,
                                                        PyArray_DIMS(%(softmaxes)s),
                                                        PyArray_DescrFromType(%(out_typenum)s), 0);
            if(!%(dx)s) {
                PyErr_SetString(PyExc_MemoryError,
                     "failed to alloc dx output");
                %(fail)s
            }
        }



        // for all examples index i is used
        for (size_t i = 0; i < PyArray_DIMS(%(y_lengths)s)[0]; ++i)
        {
            const dtype_%(softmaxes)s eps = (dtype_%(softmaxes)s)1e-9;

            // the temporal slice size for updates is given by the stride
            // length of dx along its second dimension
            npy_intp Sdx = PyArray_STRIDES(%(dx)s)[1]/sizeof(dtype_%(dx)s);
            npy_intp Ssm = PyArray_STRIDES(%(softmaxes)s)[1]/sizeof(dtype_%(softmaxes)s);

            // the distribution slice size for updates:
            npy_intp Sdx_dist = PyArray_STRIDES(%(dx)s)[2]/sizeof(dtype_%(dx)s);
            npy_intp Ssm_dist = PyArray_STRIDES(%(softmaxes)s)[2]/sizeof(dtype_%(softmaxes)s);

            // stride size for each example:
            npy_intp g_cost_stride  = PyArray_STRIDES(%(g_costs)s)[0];
            npy_intp dx_stride      = PyArray_STRIDES(%(dx)s)[0];
            npy_intp softmax_stride = PyArray_STRIDES(%(softmaxes)s)[0];
            npy_intp y_idxes_stride = PyArray_STRIDES(%(y_idxes)s)[0];
            npy_intp y_startidxes_stride = PyArray_STRIDES(%(y_startidxes)s)[0];
            npy_intp y_lengths_stride = PyArray_STRIDES(%(y_lengths)s)[0];

            npy_intp y_idxes_temp_stride = PyArray_STRIDES(%(y_idxes)s)[1]/sizeof(dtype_%(y_idxes)s);


            // slices for example i:
            dtype_%(dx) s*      __restrict__    dx_i = (dtype_%(dx)s*)(PyArray_BYTES(%(dx)s) + dx_stride * i);
            dtype_%(y_idxes) s* __restrict__ idxes_i = (dtype_%(y_idxes)s*)(PyArray_BYTES(%(y_idxes)s) + y_idxes_stride * i);
            const dtype_%(softmaxes)s* __restrict__ softmaxes_i = (dtype_%(softmaxes)s*)(PyArray_BYTES(%(softmaxes)s) + softmax_stride * i);
            const dtype_%(g_costs)s g_costs_i = ((dtype_%(g_costs)s*)(PyArray_BYTES(%(g_costs)s) + g_cost_stride * i))[0];
            const dtype_%(y_lengths) s y_lengths_i = ((dtype_%(y_lengths)s*)(PyArray_BYTES(%(y_lengths)s) + y_lengths_stride * i))[0];
            const dtype_%(y_startidxes) s y_startidxes_i = ((dtype_%(y_startidxes)s*)(PyArray_BYTES(%(y_startidxes)s) + y_startidxes_stride * i))[0];

            for (size_t j = 0 ; j < y_lengths_i; ++j)
            {
                if (idxes_i[(y_startidxes_i + j) * y_idxes_temp_stride] < 0 || idxes_i[(y_startidxes_i + j) * y_idxes_temp_stride] >= PyArray_DIMS(%(softmaxes)s)[2]) {
                    PyErr_Format(PyExc_ValueError,
                         "Softmax Index for KL Divergence is out of range ( %%ld  not in [0, %%ld]",
                         (long int)idxes_i[(y_startidxes_i + j) * y_idxes_temp_stride],
                         (long int)PyArray_DIMS(%(softmaxes)s)[2]);
                    %(fail)s;
                }
                dx_i[(y_startidxes_i + j) * Sdx + idxes_i[(y_startidxes_i + j) * y_idxes_temp_stride] * Sdx_dist] = -1. / (
                softmaxes_i[(y_startidxes_i + j) * Ssm + idxes_i[(y_startidxes_i + j) * y_idxes_temp_stride] * Ssm_dist] * g_costs_i + eps);
            }

        }
        """ % dict(locals(), **sub)

    def grad(self, *args):
        raise NotImplementedError()

masked_loss_dx = MaskedLossDx()

class MaskedLoss(gof.Op):
    nin = 3
    nout = 1
    """Masked Loss for sequence"""

    def perform(self, node, input_storage, output_storage):
        softmaxes, y_idxes, y_lengths, y_startidxes = input_storage
        prediction_cost = np.zeros(y_lengths.shape[0], dtype=softmaxes.dtype)
        # for all lengths to be predicted
        for i in range(y_lengths.shape[0]):
            # take the total cost to be the errors made
            prediction_cost[i] -= np.log(softmaxes[i,
                                                   np.arange(y_startidxes[i], y_startidxes[i] + y_lengths[i]),
                                                   y_idxes[i, y_startidxes[i] :y_startidxes[i] + y_lengths[i]]
                                                   ]).sum()

        output_storage[0][0] = prediction_cost

    def c_code(self, node, name, inp, out, sub):
        softmaxes, y_idxes, y_lengths, y_startidxes = inp
        errors, = out
        out_typenum = node.inputs[0].type.dtype_specs()[2]
        return """
        if ((PyArray_TYPE(%(softmaxes)s) != NPY_DOUBLE) &&
            (PyArray_TYPE(%(softmaxes)s) != NPY_FLOAT))
        {
            PyErr_SetString(PyExc_TypeError,
                 "softmaxes type should be float32 or float64");
            %(fail)s;
        }
        if ((PyArray_NDIM(%(softmaxes)s) != 3)
            || (PyArray_NDIM(%(y_idxes)s) != 2)
            || (PyArray_NDIM(%(y_lengths)s) != 1)
            || (PyArray_NDIM(%(y_startidxes)s) != 1))
        {
            PyErr_SetString(PyExc_ValueError, "rank error");
            %(fail)s;
        }
        if (PyArray_DIMS(%(softmaxes)s)[0] != PyArray_DIMS(%(y_lengths)s)[0])
        {
            PyErr_Format(PyExc_ValueError,
                         "softmaxes.shape[0] (%%ld) != y_lengths.shape[0] (%%ld)",
                         (long int)PyArray_DIMS(%(softmaxes)s)[0],
                         (long int)PyArray_DIMS(%(y_lengths)s)[0]);
            %(fail)s;
        }
        if (PyArray_DIMS(%(softmaxes)s)[0] != PyArray_DIMS(%(y_startidxes)s)[0])
        {
            PyErr_Format(PyExc_ValueError,
                         "softmaxes.shape[0] (%%ld) != y_startidxes.shape[0] (%%ld)",
                         (long int)PyArray_DIMS(%(softmaxes)s)[0],
                         (long int)PyArray_DIMS(%(y_startidxes)s)[0]);
            %(fail)s;
        }
        if (PyArray_DIMS(%(softmaxes)s)[0] != PyArray_DIMS(%(y_idxes)s)[0])
        {
            PyErr_Format(PyExc_ValueError,
                         "softmaxes.shape[0] (%%ld) != y_idxes.shape[0] (%%ld)",
                         (long int)PyArray_DIMS(%(softmaxes)s)[0],
                         (long int)PyArray_DIMS(%(y_idxes)s)[0]);
            %(fail)s;
        }
        if ((NULL == %(errors)s)
            || (PyArray_DIMS(%(errors)s)[0] != PyArray_DIMS(%(softmaxes)s)[0]))
        {
            if (NULL != %(errors)s) Py_XDECREF(%(errors)s);
            %(errors)s = (PyArrayObject*) PyArray_Zeros(1,
                                                        PyArray_DIMS(%(softmaxes)s),
                                                        PyArray_DescrFromType(%(out_typenum)s), 0);
            if(!%(errors)s) {
                PyErr_SetString(PyExc_MemoryError,
                     "failed to alloc errors output");
                %(fail)s
            }
        }

        // for all examples index i is used
        for (size_t i = 0; i < PyArray_DIMS(%(y_lengths)s)[0]; ++i)
        {

            // the temporal slice size for updates is given by the stride
            // length of dx along its second dimension
            npy_intp Ssm = PyArray_STRIDES(%(softmaxes)s)[1]/sizeof(dtype_%(softmaxes)s);

            // the distribution slice size for updates:
            npy_intp Ssm_dist = PyArray_STRIDES(%(softmaxes)s)[2]/sizeof(dtype_%(softmaxes)s);

            // stride size for each example:
            npy_intp error_stride  = PyArray_STRIDES(%(errors)s)[0];
            npy_intp softmax_stride = PyArray_STRIDES(%(softmaxes)s)[0];
            npy_intp y_idxes_stride = PyArray_STRIDES(%(y_idxes)s)[0];
            npy_intp y_startidxes_stride = PyArray_STRIDES(%(y_startidxes)s)[0];
            npy_intp y_lengths_stride = PyArray_STRIDES(%(y_lengths)s)[0];

            npy_intp y_idxes_temp_stride = PyArray_STRIDES(%(y_idxes)s)[1]/sizeof(dtype_%(y_idxes)s);


            // slices for example i:
            dtype_%(errors) s* __restrict__ errors_i = (dtype_%(errors)s*)(PyArray_BYTES(%(errors)s) + error_stride * i);
            dtype_%(y_idxes) s* __restrict__ idxes_i = (dtype_%(y_idxes)s*)(PyArray_BYTES(%(y_idxes)s) + y_idxes_stride * i);
            const dtype_%(softmaxes)s* __restrict__ softmaxes_i = (dtype_%(softmaxes)s*)(PyArray_BYTES(%(softmaxes)s) + softmax_stride * i);
            const dtype_%(y_lengths) s y_lengths_i = ((dtype_%(y_lengths)s*)(PyArray_BYTES(%(y_lengths)s) + y_lengths_stride * i))[0];
            const dtype_%(y_startidxes) s y_startidxes_i = ((dtype_%(y_startidxes)s*)(PyArray_BYTES(%(y_startidxes)s) + y_startidxes_stride * i))[0];

            for (size_t j = 0 ; j < y_lengths_i; ++j) {
                if (idxes_i[(y_startidxes_i + j) * y_idxes_temp_stride] < 0 || idxes_i[(y_startidxes_i + j) * y_idxes_temp_stride] >= PyArray_DIMS(%(softmaxes)s)[2]) {
                    PyErr_Format(PyExc_ValueError,
                         "Softmax Index for KL Divergence is out of range ( %%ld  not in [0, %%ld]",
                         (long int)idxes_i[(y_startidxes_i + j) * y_idxes_temp_stride],
                         (long int)PyArray_DIMS(%(softmaxes)s)[2]);
                    %(fail)s;
                }
                errors_i[0] -= log( softmaxes_i[(y_startidxes_i + j) * Ssm + idxes_i[(y_startidxes_i + j) * y_idxes_temp_stride] * Ssm_dist]);
            }

        }
        """ % dict(locals(), **sub)

    def make_node(self, softmaxes, y_idxes, y_lengths, y_startidxes, **kwargs):
        softmaxes = T.as_tensor_variable(softmaxes)
        y_idxes = T.as_tensor_variable(y_idxes)
        y_lengths = T.as_tensor_variable(y_lengths)
        y_startidxes = T.as_tensor_variable(y_startidxes)
        if (softmaxes.type.ndim != 3 or
            softmaxes.type.dtype not in T.float_dtypes):
            raise ValueError('dy must be 3-d tensor of floats', softmaxes.type)

        if (y_idxes.type.ndim != 2 or
            y_idxes.type.dtype not in T.discrete_dtypes):
            raise ValueError('y_idxes must be 2-d tensor of integers', y_idxes.type)

        if (y_lengths.type.ndim != 1 or
            y_lengths.type.dtype not in T.discrete_dtypes):
            raise ValueError('y_lengths must be 1-d tensor of integers', y_lengths.type)

        if (y_startidxes.type.ndim != 1 or
            y_startidxes.type.dtype not in T.discrete_dtypes):
            raise ValueError('y_startidxes must be 1-d tensor of integers', y_startidxes.type)

        return Apply(self, [softmaxes, y_idxes, y_lengths, y_startidxes], [
            T.Tensor(dtype=softmaxes.dtype, broadcastable=[False])()])

    def grad(self, inp, grads):
        softmaxes, y_idxes, y_lengths, y_startidxes = inp
        g_costs, = grads
        return [masked_loss_dx(softmaxes, y_idxes, y_lengths, y_startidxes, g_costs),
                grad_not_implemented(self, 1, y_idxes),
                grad_not_implemented(self, 1, y_lengths),
                grad_not_implemented(self, 1, y_startidxes)]

class MaskedSumDx(gof.Op):
    """
    Gradient of the sum of values along the third dimension
    for a 3d tensor for some subranges defined by a start dimension
    and a length along which the gradient is computed.
    """

    def make_node(self, y, y_starts, y_lengths, g_costs, **kwargs):
        y = T.as_tensor_variable(y)
        y_lengths = T.as_tensor_variable(y_lengths)
        y_starts = T.as_tensor_variable(y_starts)
        g_costs = T.as_tensor_variable(g_costs)

        if (y.type.ndim != 3 or
            y.type.dtype not in T.float_dtypes):
            raise ValueError('y must be 3-d tensor of floats', y.type)

        if (y_lengths.type.ndim != 1 or
            y_lengths.type.dtype not in T.discrete_dtypes):
            raise ValueError('y_lengths must be 1-d tensor of integers', y_lengths.type)

        if (y_starts.type.ndim != 1 or
            y_starts.type.dtype not in T.discrete_dtypes):
            raise ValueError('y_starts must be 1-d tensor of integers', y_starts.type)

        if (g_costs.type.ndim != 1 or
            g_costs.type.dtype not in T.float_dtypes):
            raise ValueError('g_costs must be 1-d tensor of floats', g_costs.type)

        return Apply(self, [y, y_starts, y_lengths, g_costs],
                     [T.Tensor(dtype=y.dtype, broadcastable=y.type.broadcastable)()])

    def perform(self, node, input_storage, output_storage):
        y, y_starts, y_lengths, g_costs = input_storage

        dx = np.zeros_like(y)
        for i in range(y_starts.shape[0]):
            # d/dx x = 1:
            dx[i, y_starts[i]:y_starts+y_lengths[i],:] = g_costs[i]

        output_storage[0][0] = dx

    def c_code_cache_version(self):
        return (3,)

    def __init__(self, **kwargs):
        gof.Op.__init__(self, **kwargs)

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return T.hashtype(self)

    def __str__(self):
        return self.__class__.__name__

    def c_code(self, node, name, inp, out, sub):
        y, y_starts, y_lengths, g_costs = inp
        dx, = out
        out_typenum = node.inputs[0].type.dtype_specs()[2]
        return """

        if ((PyArray_TYPE(%(g_costs)s) != NPY_DOUBLE) &&
            (PyArray_TYPE(%(g_costs)s) != NPY_FLOAT))
        {
            PyErr_SetString(PyExc_TypeError,
                 "g_costs type should be float32 or float64");
            %(fail)s;
        }
        if ((PyArray_TYPE(%(y)s) != NPY_DOUBLE) &&
            (PyArray_TYPE(%(y)s) != NPY_FLOAT))
        {
            PyErr_SetString(PyExc_TypeError,
                 "y type should be float32 or float64");
            %(fail)s;
        }
        if ((PyArray_NDIM(%(g_costs)s) != 1)
            || (PyArray_NDIM(%(y)s) != 3)
            || (PyArray_NDIM(%(y_starts)s) != 1)
            || (PyArray_NDIM(%(y_lengths)s) != 1))
        {
            PyErr_SetString(PyExc_ValueError, "rank error");
            %(fail)s;
        }
        if (PyArray_DIMS(%(g_costs)s)[0] != PyArray_DIMS(%(y)s)[0])
        {
            PyErr_Format(PyExc_ValueError,
                         "g_costs.shape[0] (%%ld) != y.shape[0] (%%ld)",
                         (long int)PyArray_DIMS(%(g_costs)s)[0],
                         (long int)PyArray_DIMS(%(y)s)[0]);
            %(fail)s;
        }
        if (PyArray_DIMS(%(g_costs)s)[0] != PyArray_DIMS(%(y_starts)s)[0])
        {
            PyErr_Format(PyExc_ValueError,
                         "g_costs.shape[0] (%%ld) != y_starts.shape[0] (%%ld)",
                         (long int)PyArray_DIMS(%(g_costs)s)[0],
                         (long int)PyArray_DIMS(%(y_starts)s)[0]);
            %(fail)s;
        }
        if (PyArray_DIMS(%(g_costs)s)[0] != PyArray_DIMS(%(y_lengths)s)[0])
        {
            PyErr_Format(PyExc_ValueError,
                         "g_costs.shape[0] (%%ld) != y_lengths.shape[0] (%%ld)",
                         (long int)PyArray_DIMS(%(g_costs)s)[0],
                         (long int)PyArray_DIMS(%(y_lengths)s)[0]);
            %(fail)s;
        }
        if ((NULL == %(dx)s)
            || (PyArray_DIMS(%(dx)s)[0] != PyArray_DIMS(%(y)s)[0])
            || (PyArray_DIMS(%(dx)s)[1] != PyArray_DIMS(%(y)s)[1])
            || (PyArray_DIMS(%(dx)s)[2] != PyArray_DIMS(%(y)s)[2]))
        {
            if (NULL != %(dx)s) Py_XDECREF(%(dx)s);
            %(dx)s = (PyArrayObject*) PyArray_Zeros(3,
                                                        PyArray_DIMS(%(y)s),
                                                        PyArray_DescrFromType(%(out_typenum)s), 0);
            if(!%(dx)s) {
                PyErr_SetString(PyExc_MemoryError,
                     "failed to alloc dx output");
                %(fail)s
            }
        }



        // for all examples index i is used
        for (size_t i = 0; i < PyArray_DIMS(%(y_starts)s)[0]; ++i)
        {

            // the temporal slice size for updates is given by the stride
            // length of dx along its second dimension
            npy_intp Sdx = PyArray_STRIDES(%(dx)s)[1]/sizeof(dtype_%(dx)s);

            // the distribution slice size for updates:
            npy_intp Sdx_dist = PyArray_STRIDES(%(dx)s)[2]/sizeof(dtype_%(dx)s);

            // stride size for each example:
            npy_intp g_cost_stride    = PyArray_STRIDES(%(g_costs)s)[0];
            npy_intp dx_stride        = PyArray_STRIDES(%(dx)s)[0];
            npy_intp y_starts_stride  = PyArray_STRIDES(%(y_starts)s)[0];
            npy_intp y_lengths_stride = PyArray_STRIDES(%(y_lengths)s)[0];
            size_t   y_dim_2          = PyArray_DIMS(%(y)s)[2];


            // slices for example i:
            dtype_%(dx) s*      __restrict__    dx_i = (dtype_%(dx)s*)(PyArray_BYTES(%(dx)s) + dx_stride * i);
            const dtype_%(g_costs)s g_costs_i = ((dtype_%(g_costs)s*)(PyArray_BYTES(%(g_costs)s) + g_cost_stride * i))[0];
            const dtype_%(y_lengths) s y_lengths_i = ((dtype_%(y_lengths)s*)(PyArray_BYTES(%(y_lengths)s) + y_lengths_stride * i))[0];
            const dtype_%(y_starts) s y_starts_i = ((dtype_%(y_startidxes)s*)(PyArray_BYTES(%(y_startidxes)s) + y_starts_stride * i))[0];

            for (size_t j = 0 ; j < y_lengths_i; ++j)
            {
                for (size_t k = 0; k < y_dim_2; ++k)
                {
                    dx_i[(y_starts_i + j) * Sdx + k * Sdx_dist] = g_costs_i;
                }

            }

        }
        """ % dict(locals(), **sub)

    def grad(self, *args):
        raise NotImplementedError()

masked_sum_dx = MaskedSumDx()

class MaskedSum(gof.Op):
    nin = 3
    nout = 1
    """Masked sum for sequence"""

    def make_node(self, y, y_starts, y_lengths, **kwargs):
        y = T.as_tensor_variable(y)
        y_lengths = T.as_tensor_variable(y_lengths)
        y_starts = T.as_tensor_variable(y_starts)

        if (y.type.ndim != 3 or
            y.type.dtype not in T.float_dtypes):
            raise ValueError('y must be 3-d tensor of floats', y.type)

        if (y_lengths.type.ndim != 1 or
            y_lengths.type.dtype not in T.discrete_dtypes):
            raise ValueError('y_lengths must be 1-d tensor of integers', y_lengths.type)

        if (y_starts.type.ndim != 1 or
            y_starts.type.dtype not in T.discrete_dtypes):
            raise ValueError('y_starts must be 1-d tensor of integers', y_starts.type)

        return Apply(self, [y, y_starts, y_lengths],
                     [T.Tensor(dtype=y.dtype, broadcastable=y.type.broadcastable)()])

    def perform(self, node, input_storage, output_storage):
        y, y_starts, y_lengths = input_storage

        masked_acc = np.zeros([y.shape[0]], dtype=y.dtype)
        for i in range(y_starts.shape[0]):
            # sum along row / column i
            masked_acc[i] = y[i, y_starts[i]:y_starts+y_lengths[i],:].sum()

        output_storage[0][0] = masked_acc

    def c_code(self, node, name, inp, out, sub):
        softmaxes, y_idxes, y_lengths, y_startidxes = inp
        errors, = out
        out_typenum = node.inputs[0].type.dtype_specs()[2]
        return """
        if ((PyArray_TYPE(%(softmaxes)s) != NPY_DOUBLE) &&
            (PyArray_TYPE(%(softmaxes)s) != NPY_FLOAT))
        {
            PyErr_SetString(PyExc_TypeError,
                 "softmaxes type should be float32 or float64");
            %(fail)s;
        }
        if ((PyArray_NDIM(%(softmaxes)s) != 3)
            || (PyArray_NDIM(%(y_idxes)s) != 2)
            || (PyArray_NDIM(%(y_lengths)s) != 1)
            || (PyArray_NDIM(%(y_startidxes)s) != 1))
        {
            PyErr_SetString(PyExc_ValueError, "rank error");
            %(fail)s;
        }
        if (PyArray_DIMS(%(softmaxes)s)[0] != PyArray_DIMS(%(y_lengths)s)[0])
        {
            PyErr_Format(PyExc_ValueError,
                         "softmaxes.shape[0] (%%ld) != y_lengths.shape[0] (%%ld)",
                         (long int)PyArray_DIMS(%(softmaxes)s)[0],
                         (long int)PyArray_DIMS(%(y_lengths)s)[0]);
            %(fail)s;
        }
        if (PyArray_DIMS(%(softmaxes)s)[0] != PyArray_DIMS(%(y_startidxes)s)[0])
        {
            PyErr_Format(PyExc_ValueError,
                         "softmaxes.shape[0] (%%ld) != y_startidxes.shape[0] (%%ld)",
                         (long int)PyArray_DIMS(%(softmaxes)s)[0],
                         (long int)PyArray_DIMS(%(y_startidxes)s)[0]);
            %(fail)s;
        }
        if (PyArray_DIMS(%(softmaxes)s)[0] != PyArray_DIMS(%(y_idxes)s)[0])
        {
            PyErr_Format(PyExc_ValueError,
                         "softmaxes.shape[0] (%%ld) != y_idxes.shape[0] (%%ld)",
                         (long int)PyArray_DIMS(%(softmaxes)s)[0],
                         (long int)PyArray_DIMS(%(y_idxes)s)[0]);
            %(fail)s;
        }
        if ((NULL == %(errors)s)
            || (PyArray_DIMS(%(errors)s)[0] != PyArray_DIMS(%(softmaxes)s)[0]))
        {
            if (NULL != %(errors)s) Py_XDECREF(%(errors)s);
            %(errors)s = (PyArrayObject*) PyArray_Zeros(1,
                                                        PyArray_DIMS(%(softmaxes)s),
                                                        PyArray_DescrFromType(%(out_typenum)s), 0);
            if(!%(errors)s) {
                PyErr_SetString(PyExc_MemoryError,
                     "failed to alloc errors output");
                %(fail)s
            }
        }

        // for all examples index i is used
        for (size_t i = 0; i < PyArray_DIMS(%(y_lengths)s)[0]; ++i)
        {

            // the temporal slice size for updates is given by the stride
            // length of dx along its second dimension
            npy_intp Ssm = PyArray_STRIDES(%(softmaxes)s)[1]/sizeof(dtype_%(softmaxes)s);

            // the distribution slice size for updates:
            npy_intp Ssm_dist = PyArray_STRIDES(%(softmaxes)s)[2]/sizeof(dtype_%(softmaxes)s);

            // stride size for each example:
            npy_intp error_stride  = PyArray_STRIDES(%(errors)s)[0];
            npy_intp softmax_stride = PyArray_STRIDES(%(softmaxes)s)[0];
            npy_intp y_idxes_stride = PyArray_STRIDES(%(y_idxes)s)[0];
            npy_intp y_startidxes_stride = PyArray_STRIDES(%(y_startidxes)s)[0];
            npy_intp y_lengths_stride = PyArray_STRIDES(%(y_lengths)s)[0];

            npy_intp y_idxes_temp_stride = PyArray_STRIDES(%(y_idxes)s)[1]/sizeof(dtype_%(y_idxes)s);


            // slices for example i:
            dtype_%(errors) s* __restrict__ errors_i = (dtype_%(errors)s*)(PyArray_BYTES(%(errors)s) + error_stride * i);
            dtype_%(y_idxes) s* __restrict__ idxes_i = (dtype_%(y_idxes)s*)(PyArray_BYTES(%(y_idxes)s) + y_idxes_stride * i);
            const dtype_%(softmaxes)s* __restrict__ softmaxes_i = (dtype_%(softmaxes)s*)(PyArray_BYTES(%(softmaxes)s) + softmax_stride * i);
            const dtype_%(y_lengths) s y_lengths_i = ((dtype_%(y_lengths)s*)(PyArray_BYTES(%(y_lengths)s) + y_lengths_stride * i))[0];
            const dtype_%(y_startidxes) s y_startidxes_i = ((dtype_%(y_startidxes)s*)(PyArray_BYTES(%(y_startidxes)s) + y_startidxes_stride * i))[0];

            for (size_t j = 0 ; j < y_lengths_i; ++j)
            {
                errors_i[0] -= log( softmaxes_i[(y_startidxes_i + j) * Ssm + idxes_i[(y_startidxes_i + j) * y_idxes_temp_stride] * Ssm_dist]);
            }

        }
        """ % dict(locals(), **sub)

    def grad(self, inp, grads):
        y, y_starts, y_lengths, = inp
        g_costs, = grads
        return [masked_sum_dx(y, y_starts, y_lengths, g_costs),
                grad_not_implemented(self, 1, y_starts),
                grad_not_implemented(self, 1, y_lengths)]

masked_loss = MaskedLoss()
