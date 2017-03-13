from __future__ import division
import numpy as np
import tensorflow as tf

from ..base import struct, scoped_variable, \
    _default_value, _validate_axes, _get_existing_vars

from .ops import linear, expand_dims


def _validate_image(X, one_dim=False):
    shape = X.get_shape()
    assert shape.ndims == 4, shape
    if one_dim:
        assert shape[2].value == 1, shape


def _fix_stride(param):
    if len(param['stride']) == 2:
        sw, sh = param['stride']
        param['stride'] = (1, sw, sh, 1)


def _constant_pad(X, ker_shape):
    """
    Padding type `padding='CONSTANT'` for `tf.nn.conv2d`.
    Instead of padding `X` with zeros, like `SAME` does,
    use the values on the edges.
    """
    _validate_image(X)

    kw, kh = ker_shape[:2]
    wb, wa = kw // 2, kw - kw // 2 - 1
    hb, ha = kh // 2, kh - kh // 2 - 1

    X_ = tf.concat([
        tf.tile(X[:, :1], (1, wb, 1, 1)),
        X,
        tf.tile(X[:, -1:], (1, wa, 1, 1)),
    ], axis=1)
    X_pad = tf.concat([
        tf.tile(X_[:, :, :1], (1, 1, hb, 1)),
        X_,
        tf.tile(X_[:, :, -1:], (1, 1, ha, 1)),
    ], axis=1)

    return X_pad


def _get_kernel(name, scope_name, ker_shape):
    return scoped_variable(
        'kernel_%s' % name, scope_name, shape=ker_shape,
        initializer=tf.contrib.layers.xavier_initializer_conv2d()
    )


def pool(X, param, scope_name='pool'):
    """
    Pooling:
    `X` has shape `[B, W, H, C_in]`.
    `params['type']` is a string "max" or "mean".
    `params['kernel']` is a tuple `(kw, kh)`.
    `params['stride']` is `(stride_w, stride_h)` and defaults to `param['kernel']`.
    `params['pad']` is one of `SAME` (default), `VALID`.
    """
    _default_value(param, 'stride', param['kernel'])
    _default_value(param, 'pad', 'SAME')

    pool_func = \
        tf.contrib.layers.avg_pool2d if param.get('pool') == 'mean' \
        else tf.contrib.layers.max_pool2d

    kernel, stride = param['kernel'], param['stride']

    return pool_func(X, kernel, stride, param['pad'], scope=scope_name)


def conv(X, param, name, scope_name='conv'):
    """
    Convolution:
    `X` has shape `[B, W, H, C_in]`.
    `params['kernel']` is a tuple `(kw, kh, C_out)`.
    `params['stride']` is `(1, stride_w, stride_h, 1)` and defaults to `(1, 1, 1, 1)`.
    `params['pad']` is one of `SAME` (default), `VALID`, `CONSTANT`.
    """
    _default_value(param, 'stride', (1, 1, 1, 1))
    _default_value(param, 'pad', 'SAME')
    _default_value(param, 'rate', 1)
    _fix_stride(param)

    kw, kh, c_out = param['kernel']
    c_in = X.get_shape()[3].value
    ker_shape = (kw, kh, c_in, c_out)

    pad_type = param['pad']
    if pad_type == 'CONSTANT':
        assert param['stride'] == (1, 1, 1, 1)
        X = _constant_pad(X, ker_shape)
        pad_type = 'VALID'

    kernel = _get_kernel(name, scope_name, ker_shape)

    if param['rate'] == 1:
        conv = tf.nn.conv2d(
            X, kernel, param['stride'],
            padding=pad_type, name='conv_%s' % name
        )

    else:
        assert param['rate'] > 1, "Rate must be >= 1."
        assert param['stride'] == (1, 1, 1, 1), "Cannot mix strides and rates."

        conv = tf.nn.atrous_conv2d(
            X, kernel, param['rate'], pad_type,
            name='atrous_conv_%s' % name
        )

    if param.get('bias'):
        bias_shape = (1, 1, 1, c_out)
        conv += scoped_variable(
            'ker_bias_%s' % name, scope_name,
            shape=bias_shape,
            initializer=tf.zeros_initializer,
        )

    if param.get('pool'):
        conv = pool(conv, param['pool'])

    return conv


def deconv(X, param, name, scope_name='deconv'):
    """
    Deconvolution:
    `X` has shape `[B, W, H, C_in]`.

    `param['kernel']` is a tuple `(kw, kh, C_out, C_in)` or `(kw, kh, C_out)`.
    If `C_in` is not given, then it must be inferrable from `X.get_shape()`.

    param['stride'] is a tuple `(1, stride_w, stride_h, 1)`, defaults to `(1, 1, 1, 1)`
    param['pad'] is one of `SAME` (default), `VALID`.
    """
    _validate_image(X)
    _default_value(param, 'stride', (1, 1, 1, 1))
    _default_value(param, 'pad', 'SAME')
    _fix_stride(param)

    c_in = X.get_shape()[3].value
    ker_shape = param['kernel']

    if c_in:
        if len(ker_shape) == 3:
            kw, kh, c_out = param['kernel']
        else:
            kw, kh, c_out, c_in_ = param['kernel']
            assert c_in_ == c_in, \
                "Inferred and given numbers of input_channels do not agree."

    else:
        if len(ker_shape) == 4:
            kw, kh, c_out, c_in = param['kernel']
        else:
            assert False, \
                "Number of input_channels was not given and could not be inferred."

    ker_shape = (kw, kh, c_out, c_in)
    kernel = _get_kernel(name, scope_name, ker_shape)

    pad_type = param['pad']
    b, w, h, _ = X.get_shape().as_list()
    if w and h:
        _, sw, sh, _ = param['stride']
        w, h = w * sw, h * sh
        if pad_type == 'VALID':
            w += kw - 1
            h += kh - 1

    input_shape = tf.shape(X)
    wh_dims = input_shape[1:3] * param['stride'][1:3]
    if pad_type == 'VALID':
        wh_dims += param['kernel'][:2]
        wh_dims -= 1

    output_shape = tf.concat([
        input_shape[:1],
        wh_dims,
        param['kernel'][2:3]
    ], axis=0)

    deconv = tf.nn.conv2d_transpose(X, kernel,
                                    output_shape, param['stride'],
                                    name='deconv_%s' % name,
                                    padding=pad_type)

    if param.get('bias'):
        deconv += scoped_variable(
            'bias_%s' % name, scope_name,
            shape=(1, 1, 1, c_out),
            initializer=tf.zeros_initializer,
        )

    if w and h:
        deconv.set_shape([b, w, h, c_out])

    return deconv


def conv_1x1(X, c_out, name, scope_name, bias=False):
    """
    Perform a 1x1 convolution.
    If `X` is 2-dimensional, then just perform a matrix-multiplication.
    """
    dim = X.get_shape().ndims
    if dim == 2:
        X = expand_dims(X, [1, 2])

    param = {
        'kernel': (1, 1, c_out),
        'bias': bias,
    }
    XX = conv(X, param, name, scope_name)
    if dim == 2:
        XX = tf.squeeze(XX, [1, 2])
    return XX
