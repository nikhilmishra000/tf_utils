from __future__ import division
import cPickle as pickle
import os.path as osp
import numpy as np
import tensorflow as tf


def make_session(frac):
    """ Create a tf.Session(), limiting fraction of gpu that is allocated. """
    return tf.Session(config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=frac)
    ))


def make_placeholders(variables):
    """ Shortcut to make placeholders. Default dtype is tf.float32.
    Input: a dict{name: shape} or {name: (shape, dtype_str)}
           <shape> is a tuple of ints
           <dtype_str> is a str like 'tf.float32', 'tf.int32', etc
    Usage:
         for var in make_placeholders(variables):
             exec(var)
    """
    commands = []
    for name, args in variables.items():
        if len(args) == 2 and isinstance(args[1], basestring):
            shape, dtype = args
        else:
            shape, dtype = args, 'tf.float32'
        commands.append(
            "{0} = tf.placeholder({2}, {1})".format(name, shape, dtype)
        )
    return commands


def affine(X, dim_out, scope=None, tag='', reuse=None):
    """ Affine: Xw + b
        X has shape [batch, dim_in], 
        w is [dim_in, dim_out], b is [1, dim_out]
    """
    if scope is None:
        scope = tf.VariableScope(reuse, name='affine',
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))

    with tf.variable_scope(scope):
        dim_in = X.get_shape()[1]
        W = tf.get_variable("w_{0}".format(tag),
                            shape=(dim_in, dim_out), dtype=tf.float32)
        b = tf.get_variable("b_{0}".format(tag),
                            shape=(dim_out,), dtype=tf.float32)
    return tf.nn.xw_plus_b(X, W, b)


def conv(X, param, scope=None, tag='', reuse=None):
    """ Convolution:
        X has shape [batch, width, height, in_channels],
        params['kernel'] has shape [kernel_w, kernel_h, in_channels, out_channels]
        params['stride'] is (1, stride_w, stride_h, 1) and defaults to (1,1,1,1)
        params['pad'] is either "SAME" or "VALID" (defaults to the latter)
    """
    if 'stride' not in param:
        param['stride'] = (1, 1, 1, 1)
    if 'pad' not in param:
        param['pad'] = "VALID"

    if scope is None:
        scope = tf.VariableScope(reuse, name='conv_' + str(tag),
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))

    with tf.variable_scope(scope):
        kernel = tf.get_variable('kernel_{0}'.format(tag), param['kernel'])
        conv = tf.nn.conv2d(X, kernel, param['stride'], padding=param['pad'])
    return conv


def deconv(X, param, scope, tag=''):
    """ Deconvolution:
        X has shape [batch, width, height, in_channels]
        param['kernel'] has shape [kernel_w, kernel_h, out_channels, in_channels]
        param['output'] has shape [new_width, new_height, out_channels]
        params['stride'] is (1, stride_w, stride_h, 1) and defaults to (1,1,1,1)
        params['pad'] is either "SAME" or "VALID" (defaults to the former)
    """
    if 'stride' not in param:
        param['stride'] = (1, 1, 1, 1)
    if 'pad' not in param:
        param['pad'] = 'SAME'

    with tf.variable_scope(scope):
        kernel = tf.get_variable('kernel_{0}'.format(tag), param['kernel'])
        deconv = tf.nn.conv2d_transpose(X, kernel,
                                        param['output'], param['stride'],
                                        name='deconv_' + str(tag))
    return deconv


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


def channel_softmax(X):
    """ Channel softmax:
        X has shape [batch, width, height, channels],
        Return Y = e.^X such that Y[b,w,h,:] sums to 1.
    """
    X = tf.exp(X)
    return X / tf.reduce_sum(X, reduction_indices=[3], keep_dims=True)


def norm(X, axis=0, keepdims=False, p=2):
    if isinstance(axis, int):
        axis = [axis]
    assert isinstance(axis, list)
    return tf.pow(
        tf.reduce_sum(tf.pow(X, p), axis, keepdims),
        1.0 / p
    )


def L1_loss(pred, target, weights=1):
    return tf.reduce_sum(
        tf.abs(pred - target) * weights
    )


def L2_loss(pred, target, weights=1):
    return tf.reduce_sum(
        tf.square(pred - target) * weights
    )


class struct(dict):

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self
