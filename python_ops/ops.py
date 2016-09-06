from __future__ import division
import cPickle as pickle
import os.path as osp
import numpy as np
import tensorflow as tf

from ..base import scoped_variable, \
    _default_value, _validate_axes


def affine(X, dim_out, name='', scope_name='affine'):
    """ Affine: X*W + b
        X has shape [batch, dim_in]
        Then W, b will be [dim_in, dim_out], [1, dim_out]
    """
    assert X.get_shape().ndims == 2
    dim_in = X.get_shape()[1]
    W = scoped_variable('w_%s' % name,
                        scope_name,
                        shape=(dim_in, dim_out), dtype=tf.float32,
                        initializer=tf.contrib.layers.xavier_initializer())
    b = scoped_variable('b_%s' % name,
                        scope_name,
                        shape=(dim_out,), dtype=tf.float32)

    return tf.nn.xw_plus_b(X, W, b, 'affine_%s' % name)


def conv(X, param, name, scope_name='conv'):
    """ Convolution:
        X has shape [batch, width, height, in_channels],
        params['kernel'] has shape [kernel_w, kernel_h, in_channels, out_channels]
        params['stride'] is (1, stride_w, stride_h, 1) and defaults to (1,1,1,1)
        params['pad'] is either "SAME" or "VALID" (defaults to the SAME)
    """
    assert X.get_shape().ndims == 4

    _default_value(param, 'stride', (1, 1, 1, 1))
    _default_value(param, 'pad', 'SAME')

    kernel = scoped_variable('kernel_%s' % name, scope_name,
                             shape=param['kernel'],
                             initializer=tf.contrib.layers.xavier_initializer())
    conv = tf.nn.conv2d(X, kernel, param['stride'], padding=param['pad'],
                        name='conv_%s' % name)
    return conv


def deconv(X, param, name, scope_name='deconv'):
    """ Deconvolution:
        X has shape [batch, width, height, in_channels]
        param['kernel'] is a tuple(kernel_w, kernel_h, out_channels, in_channels)
        params['stride'] is a tuple(1, stride_w, stride_h, 1), defaults to (1,1,1,1)
        params['pad'] is either "SAME" or "VALID" (defaults to SAME)
    """
    assert X.get_shape().ndims == 4

    _default_value(param, 'stride', (1, 1, 1, 1))
    _default_value(param, 'pad', 'SAME')

    kernel = scoped_variable('kernel_%s' % name, scope_name,
                             shape=param['kernel'])

    input_shape = tf.shape(X)

    wh_dims = input_shape[1:3] * param['stride'][1:3]

    if param['pad'] == 'VALID':
        wh_dims += param['kernel'][:2]
        wh_dims -= 1

    output_shape = tf.concat(0, [
        input_shape[:1],
        wh_dims,
        param['kernel'][2:3]
    ])

    deconv = tf.nn.conv2d_transpose(X, kernel,
                                    output_shape, param['stride'],
                                    name='deconv_%s' % name,
                                    padding=param['pad'])
    return deconv


def make_stack(func):
    def generic_stack(X, params, nonlin, name,
                      raw_output=True,
                      initializer=tf.truncated_normal_initializer(stddev=0.1)):
        if not isinstance(nonlin, list):
            nonlin = [nonlin] * len(params)
        for i, param in enumerate(params):
            X = func(X, param, i, name)
            if not raw_output or i + 1 < len(params):
                X = nonlin[i](X)
        return X
    return generic_stack

fc_stack = make_stack(affine)
conv_stack = make_stack(conv)
deconv_stack = make_stack(deconv)


def spatial_softmax(X):
    """ Spatial softmax:
        X has shape [batch, width, height, channels],
        each channel defines a spatial distribution,
        taking expectation gives pairs (x,y) of feature points.
        Output has shape [channels, 2].
    """
    _, w, h, _ = X.get_shape()
    x_map, y_map = tf.linspace(0., 1., w), tf.linspace(0., 1., h)
    x_map, y_map = tf.reshape(x_map, (1, w.value, 1)
                              ), tf.reshape(y_map, (1, h.value, 1))

    X = tf.exp(X)
    fx, fy = tf.reduce_sum(X, [1]), tf.reduce_sum(X, [2])
    fx /= tf.reduce_sum(fx, [1], keep_dims=True)
    fy /= tf.reduce_sum(fy, [1], keep_dims=True)
    fx = tf.reduce_sum(fx * x_map, [1])
    fy = tf.reduce_sum(fy * y_map, [1])

    return tf.concat(1, [fx, fy])


def norm(X, axis=0, keep_dims=False, p=2):
    """
    Compute the norm of a tensor across the given axes.
    Like np.linalg.norm.

    :param X: a Tensor of arbitrary dimensions
    :param axis: an int or list(int) of axes <= ndim(X)
    :param keep_dims: bool
    :param p: float > 0
    """
    axis = _validate_axes(axis)
    return tf.pow(
        tf.reduce_sum(tf.pow(X, p), axis, keep_dims),
        1.0 / p
    )


def normalize(X, axis):
    """
    Normalize a Tensor so that it sums to one across the given axis.
    """
    axis = _validate_axes(axis)
    return X / tf.reduce_sum(X, axis, keep_dims=True)


def softmax(X, axis):
    """
    Compute a softmax across arbitrary dimensions.
    """
    return normalize(tf.exp(X), axis)
