from __future__ import division
import numpy as np
import tensorflow as tf

from ..base import \
    scoped_variable, _validate_axes


def affine(X, dim_out, name='', scope_name='affine'):
    """ Affine: X*W + b
        X has shape [batch, dim_in]
        Then W, b will be [dim_in, dim_out], [1, dim_out]
    """
    assert X.get_shape().ndims == 2
    dim_in = X.get_shape()[1]
    W = scoped_variable(
        'w_%s' % name, scope_name,
        shape=(dim_in, dim_out), dtype=X.dtype,
        initializer=tf.contrib.layers.xavier_initializer()
    )
    b = scoped_variable(
        'b_%s' % name, scope_name,
        shape=(dim_out,), dtype=X.dtype,
        initializer=tf.zeros_initializer,
    )

    return tf.nn.xw_plus_b(X, W, b, 'affine_%s' % name)


def linear(X, dim_out, name='', scope_name='linear'):
    """ Linear: X*W
        X has shape [batch, dim_in]
        Then W, b will be [dim_in, dim_out], [1, dim_out]
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
    shape = X.get_shape().as_list()
    assert None not in shape[keep_first_k:]

    d = np.prod(shape[keep_first_k:])
    new_shape = [s if s else -1 for s in shape[:keep_first_k] + [d]]
    assert new_shape.count(-1) <= 1

    return tf.reshape(X, new_shape)


def batch_norm(X, param, name, update=True):
    axis = _validate_axes(param['axis'])
    mu, sigsq = tf.nn.moments(X, axis)

    shape = [1 if d in axis else d
             for d in X.get_shape().as_list()]
    mean = scoped_variable('mean', name, initial_value=tf.zeros(shape))
    std = scoped_variable('var', name, initial_value=tf.ones(shape))

    if not update:
        normed = (X - mean) / std

    else:
        g = param['alpha']
        new_mean = (1 - g) * mean + g * mu
        new_std = (1 - g) * std + g * tf.sqrt(sigsq)

        normed = X - new_mean
        if 'gamma' in param:
            normed *= param['gamma']

        eps = param.get('epsilon', 1e-8)
        normed /= (new_std + eps)
        if 'beta' in param:
            normed += param['beta']

        tf.add_to_collection(
            tf.assign(mean, new_mean), tf.GraphKeys.UPDATE_OPS
        )
        tf.add_to_collection(
            tf.assign(std, new_std), tf.GraphKeys.UPDATE_OPS
        )

    return normed
