from __future__ import division
import numpy as np
import tensorflow as tf

from ..base import scoped_variable, \
    _default_value, _validate_axes


def _constant_pad(X, ker_shape):
    """
    Padding type `padding='CONSTANT'` for `tf.nn.conv2d`.
    Instead of padding `X` with zeros, like `SAME` does,
    use the values on the edges.
    """
    assert X.get_shape().ndims == 4
    assert len(ker_shape) == 4

    kw, kh = ker_shape[:2]
    wb, wa = kw // 2, kw - kw // 2 - 1
    hb, ha = kh // 2, kh - kh // 2 - 1

    X_ = tf.concat(1, [
        tf.tile(X[:, :1], (1, wb, 1, 1)),
        X,
        tf.tile(X[:, -1:], (1, wa, 1, 1)),
    ])

    X_pad = tf.concat(2, [
        tf.tile(X_[:, :, :1], (1, 1, hb, 1)),
        X_,
        tf.tile(X_[:, :, -1:], (1, 1, ha, 1)),
    ])

    return X_pad


def conv(X, param, name, scope_name='conv'):
    """
    Convolution:
    `X` has shape `[B, W, H, C_in]`.
    `params['kernel']` is a tuple `(kw, kh, C_out)`.
    `params['stride']` is `(1, stride_w, stride_h, 1)` and defaults to `(1, 1, 1, 1)`.
    `params['pad']` is one of `SAME` (default), `VALID`, `CONSTANT`.
    """
    assert X.get_shape().ndims == 4

    _default_value(param, 'stride', (1, 1, 1, 1))
    _default_value(param, 'pad', 'SAME')
    _default_value(param, 'rate', 1)

    kw, kh, c_out = param['kernel']
    c_in = X.get_shape()[3].value

    ker_shape = (kw, kh, c_in, c_out)
    pad_type = param['pad']
    if pad_type == 'CONSTANT':
        assert param['stride'] == (1, 1, 1, 1)
        X = _constant_pad(X, ker_shape)
        pad_type = 'VALID'

    kernel = scoped_variable(
        'kernel_%s' % name, scope_name,
        shape=ker_shape,
        initializer=tf.contrib.layers.xavier_initializer_conv2d()
    )

    if len(param['stride']) == 2:
        sw, sh = param['stride']
        param['stride'] = (1, sw, sh, 1)

    if param['rate'] == 1:
        conv = tf.nn.conv2d(
            X, kernel, param['stride'],
            padding=pad_type, name='conv_%s' % name
        )

    else:
        assert param['rate'] > 1, \
            "Rate must be >= 1."
        assert param['stride'] == (1, 1, 1, 1), \
            "Cannot mix strides and rates."

        conv = tf.nn.atrous_conv2d(
            X, kernel, param['rate'], pad_type,
            name='atrous_conv_%s' % name
        )

    if param.get('bias'):
        bias_shape = (1, 1, 1, c_out)
        conv += scoped_variable(
            'bias_%s' % name, scope_name,
            shape=bias_shape,
            initializer=tf.zeros_initializer,
        )

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
    assert X.get_shape().ndims == 4

    _default_value(param, 'stride', (1, 1, 1, 1))
    _default_value(param, 'pad', 'SAME')

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
    kernel = scoped_variable(
        'kernel_%s' % name, scope_name,
        shape=ker_shape,
        initializer=tf.contrib.layers.xavier_initializer_conv2d()
    )

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

    output_shape = tf.concat(0, [
        input_shape[:1],
        wh_dims,
        param['kernel'][2:3]
    ])

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


def causal_conv(X, param, name, scope):
    """
    Causal/atrous convolution like WaveNet.

    `X` has shape `[B, T, 1, C]`.
    `param['kernel']` is a `tuple(k, 1, channels_out)`.
    `param['rate']` is an int.

    Example:
    ```
    X_pl = tf.placeholder(tf.float32, (B, T, 1, K))
    X = tfu.casual_init(X_pl)
    for i, param in enumerate(params):
        X = tfu.casual_conv(X, param, i, 'casual_conv')
    ```
    """
    assert X.get_shape().ndims == 4

    b, t, l, c_in = X.get_shape().as_list()
    k, _, c_out = param['kernel']
    rate = param.get('rate', 1)
    out_shape = (b, t, l, c_out)

    XX = tf.pad(X, [
        (0, 0), (k + (rate - 1) * (k - 1) - 1, 0),
        (0, 0), (0, 0)
    ])

    param['pad'] = 'VALID'
    XX = conv(XX, param, name, scope)
    XX.set_shape(out_shape)

    return XX


def causal_init(X):
    """
    Do some padding/shifting to `X` so that causal stuff aligns nicely.
    Should call at begining of stack, before first call to `casual_conv()`.
    See `tfu.causal_conv()` for usage.
    """
    XX = tf.pad(X[:, :-1], [(0, 0), (1, 0), (0, 0), (0, 0)])
    return XX


def spatial_softmax(X):
    """ Spatial softmax:
        X has shape [batch, width, height, channels],
        each channel defines a spatial distribution,
        taking expectation gives pairs(x, y) of feature points.
        Output has shape[channels, 2].
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
