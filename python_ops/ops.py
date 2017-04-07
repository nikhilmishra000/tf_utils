from __future__ import division
import numpy as np
import tensorflow as tf

from ..base import \
    scoped_variable, _validate_axes

xavier_init = tf.contrib.layers.xavier_initializer()


def affine(X, dim_out, name='', scope_name='affine',
           initializer=xavier_init):
    """
    Affine layer: X*W + b
    X has shape [batch, dim_in]
    Then W, b will be [dim_in, dim_out], [1, dim_out]
    """
    assert X.get_shape().ndims == 2
    dim_in = X.get_shape()[1]
    W = scoped_variable(
        'w_%s' % name, scope_name,
        shape=(dim_in, dim_out), dtype=X.dtype,
        initializer=initializer
    )
    b = scoped_variable(
        'b_%s' % name, scope_name,
        shape=(dim_out,),
        initializer=tf.zeros_initializer(X.dtype),
    )

    return tf.nn.xw_plus_b(X, W, b, 'affine_%s' % name)


def linear(X, dim_out, name='', scope_name='linear'):
    """
    Like `tfu.affine` but no bias.
    """
    assert X.get_shape().ndims == 2
    dim_in = X.get_shape()[1]
    W = scoped_variable(
        'w_%s' % name, scope_name,
        shape=(dim_in, dim_out), dtype=X.dtype,
        initializer=tf.contrib.layers.xavier_initializer()
    )

    return tf.matmul(X, W, name='linear_%s' % name)


def make_stack(func):
    def generic_stack(X, params, nonlin, name, raw_output=True):
        if not isinstance(nonlin, list):
            nonlin = [nonlin] * len(params)
        for i, param in enumerate(params):
            X = func(X, param, i, name)
            if not raw_output or i + 1 < len(params):
                X = nonlin[i](X)
        return X
    return generic_stack


def norm(X, axis=None, keep_dims=False, p=2, root=True):
    """
    Compute the norm of a tensor across the given axes.
    Like np.linalg.norm.

    :param X: a Tensor of arbitrary dimensions
    :param axis: an int, list(int), or None
    :param keep_dims: bool
    :param p: float > 0
    :param root: if False, don't take the p-th root.
    """
    axis = _validate_axes(axis)
    Y = tf.reduce_sum(tf.pow(X, p), axis, keep_dims)
    if root:
        return tf.pow(Y, 1.0 / p)
    else:
        return Y


def normalize(X, axis):
    """
    Normalize a Tensor so that it sums to one across the given axis.
    """
    axis = _validate_axes(axis)
    return X / tf.reduce_sum(X, axis, keep_dims=True)


def ravel(X, keep_first_k=1):
    """
    Flatten a tensor with shape `[d_1, ..., d_n]`
    to one of shape `[d_1, ..., d_k, D]`,
    where `D = d_{k+1} * ... * d_n`.
    """
    shape = X.get_shape().as_list()
    assert keep_first_k <= len(shape)
    assert None not in shape[keep_first_k:]

    d = np.prod(shape[keep_first_k:])
    new_shape = [s if s else -1 for s in shape[:keep_first_k] + [d]]
    assert new_shape.count(-1) <= 1

    return tf.reshape(X, new_shape)


def spatial_softmax(X):
    """
    Spatial softmax: X has shape[batch, width, height, channels].

    Interpret each channel as unnormalized log PMF values
    over the [width, height] spatial dimensions.

    Return a list of points representing the expectation of each distribution,
    output has shape [batch, 2 * channels].
    """
    _, w, h, _ = X.get_shape()
    x_map, y_map = tf.linspace(0., 1., w), tf.linspace(0., 1., h)
    x_map = tf.reshape(x_map, (1, w.value, 1))
    y_map = tf.reshape(y_map, (1, h.value, 1))

    X = tf.exp(X)
    fx, fy = tf.reduce_sum(X, [1]), tf.reduce_sum(X, [2])
    fx /= tf.reduce_sum(fx, [1], keep_dims=True)
    fy /= tf.reduce_sum(fy, [1], keep_dims=True)
    fx = tf.reduce_sum(fx * x_map, [1])
    fy = tf.reduce_sum(fy * y_map, [1])

    return tf.concat([fx, fy], axis=1)


def expand_dims(X, axes):
    """
    Like `tf.expand_dims()` but better if you want to expand multiple axes.

    :param axes: list(int)
    """
    if isinstance(axes, int):
        return tf.expand_dims(X, axes)

    else:
        shape = [tf.shape(X)[i] if dim is None else dim
                 for i, dim in enumerate(X.get_shape().as_list())]
        for ax in axes:
            shape.insert(ax, 1)
        return tf.reshape(X, shape)
