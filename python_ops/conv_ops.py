from __future__ import division
from cached_property import cached_property
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


def _get_kernel(name, scope_name, ker_shape):
    return scoped_variable(
        'kernel_%s' % name, scope_name, shape=ker_shape,
        initializer=tf.contrib.layers.xavier_initializer_conv2d()
    )


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


class CausalConv1D(object):
    """
    Implements a one-dimensional dilated convolution layer.
    Efficiently perform rollouts using a tf.FIFOQueue().
    """

    def __init__(self, params, name, scope):
        _default_value(params, 'nonlin', 'gated')
        _default_value(params, 'rate', 1)
        params['pad'] = 'VALID'
        ker_shape = params['kernel']
        assert len(ker_shape) == 3 and ker_shape[1] == 1
        self.params, self.scope = params, scope
        self.name = name

    def make_queue(self, xt, Z=None):
        B = tf.shape(xt)[0]
        rate = self.params['rate']
        k, _, c_out = self.params['kernel']
        c_in = xt.get_shape()[1].value

        queues = [
            tf.FIFOQueue(
                rate, dtypes=xt.dtype,
                name='queue_%d_%s' % (i, self.name)
            )
            for i in range(k - 1)
        ]

        def body(size):
            q.dequeue()
            return q.size()

        sizes = [q.size() for q in queues]
        flush_ops = [
            tf.while_loop(lambda size: size > 0, body, [size])
            for size in sizes
        ]

        fill_ops = [
            q.enqueue_many(tf.zeros((rate, B, c_in))) for q in queues
        ]
        X, x_to_push, push_ops = [xt], xt, []
        for i in range(k - 1):
            x_popped = queues[i].dequeue()
            x_popped.set_shape((None, c_in))
            push_ops.append(queues[i].enqueue([x_to_push]))
            x_to_push = x_popped
            X.append(x_popped)

        X_next, X_skip = self(X, Z, conv=False)

        ops = struct(
            fill=tf.group(*fill_ops),
            push=tf.group(*push_ops),
            flush=tf.group(*flush_ops),
        )

        return X_next, X_skip, ops

    def __call__(self, X_in, Z=None, conv=True):
        if isinstance(X_in, list):
            X = expand_dims(tf.pack(X_in, axis=1), axes=2)
        else:
            X = X_in

        _validate_image(X, one_dim=True)
        c_in, c_out = X.get_shape()[-1].value, self.params['kernel'][-1]

        if self.params['nonlin'] == 'gated':
            if conv:
                xf = self._do_conv(X, 'xf_%s' % self.name)
                xg = self._do_conv(X, 'xg_%s' % self.name)
            else:
                wx_f = self.get_weight('kernel_xf_%s')
                wx_g = self.get_weight('kernel_xg_%s')
                xf = tf.einsum('btlu,tluv->bv', X, wx_f)
                xg = tf.einsum('btlu,tluv->bv', X, wx_g)

            if Z is not None:
                zf = linear(Z, c_out, 'zf_%s' % self.name, self.scope)
                zg = linear(Z, c_out, 'zg_%s' % self.name, self.scope)
                if conv:
                    zf = expand_dims(zf, [1, 2])
                    zg = expand_dims(zg, [1, 2])
                xf += zf
                xg += zg

            XX = tf.tanh(xf) * tf.sigmoid(xg)

        elif self.params['nonlin'] == 'relu':
            if conv:
                XX = tf.nn.relu(self._do_conv(X, 'x_%s' % self.name))
            else:
                wx = self.get_weight('kernel_x_%s')
                XX = tf.nn.relu(tf.einsum('btlu,tluv->bv', X, wx))
        else:
            raise ValueError(self.params['nonlin'])

        if self.params.get('resid'):
            if conv:
                X += conv_1x1(
                    XX, c_in, 'xr_%s' % self.name, self.scope
                )
                X_skip = conv_1x1(
                    XX, c_in, 'xs_%s' % self.name, self.scope
                )
            else:
                wx_r = tf.squeeze(self.get_weight('kernel_xr_%s'), [0, 1])
                wx_s = tf.squeeze(self.get_weight('kernel_xs_%s'), [0, 1])
                X = X_in[-1] + tf.matmul(XX, wx_r)
                X_skip = tf.matmul(XX, wx_s)

        else:
            X, X_skip = XX, None

        return X, X_skip

    def _do_conv(self, X, name):
        k, _, c_out = self.params['kernel']
        rate = self.params['rate']

        B, T, _, c_in = X.get_shape().as_list()
        out_shape = (B, T, 1, c_out)

        X = tf.pad(X, [
            (0, 0), (k + (rate - 1) * (k - 1) - 1, 0),
            (0, 0), (0, 0)
        ])
        X = conv(X, self.params, name, self.scope)
        X.set_shape(out_shape)

        return X

    def get_weight(self, name):
        return self.weights[(name % self.name, self.scope)]

    @cached_property
    def weights(self):
        if self.params['nonlin'] == 'gated':
            weights = _get_existing_vars([
                ('kernel_xf_%s' % self.name, self.scope),
                ('kernel_xg_%s' % self.name, self.scope),
                ('kernel_xr_%s' % self.name, self.scope),
                ('kernel_xs_%s' % self.name, self.scope),
            ])

        elif self.params['nonlin'] == 'relu':
            weights = _get_existing_vars([
                ('kernel_x_%s' % self.name, self.scope),
                ('kernel_xr_%s' % self.name, self.scope),
                ('kernel_xs_%s' % self.name, self.scope),
            ])

        else:
            raise ValueError(self.params['nonlin'])

        assert len(weights) > 0, \
            "Weights have not been created yet"

        return weights
