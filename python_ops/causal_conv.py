from __future__ import division
from cached_property import cached_property
import numpy as np
import tensorflow as tf

from ..base import struct, scoped_variable, \
    _default_value, _validate_axes, _get_existing_vars

from .ops import linear, expand_dims

from .conv_ops import _validate_image, conv, conv_1x1


def causal_conv(X, param, name, scope):
    """
    Original dilated conv implementation for backwards compatibility.
    """
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


class CausalConv1D(object):
    """
    Implements an efficient one-dimensional dilated convolution layer.

    For training:
    * Use a single convolution pass.
    For sequential generations:
    * Use tf.FIFOQueues to avoid recomputing intermediate activations.
    For graph-creation rollouts:
    * Use an RNNCell-like interface in `self.cell`.
    """

    def __init__(self, params, name, scope):
        _default_value(params, 'nonlin', 'gated')
        _default_value(params, 'rate', 1)
        params['pad'] = 'VALID'
        ker_shape = params['kernel']
        assert len(ker_shape) == 3 and ker_shape[1] == 1
        self.params, self.scope = params, scope
        self.name = name

    @cached_property
    def cell(self):
        k, _, c_out = self.params['kernel']
        rate = self.params['rate']
        c_in = self.weights.values()[0].get_shape()[2].value
        parent = self

        class CausalConvCell(object):

            def zero_state(self, B):
                return [
                    [tf.zeros((B, c_in)) for _ in range(rate)]
                    for i in range(k - 1)
                ]

            def __call__(self, queues, xt, Z=None):
                X, x_to_push = [xt], xt
                for i in range(k - 1):
                    x_popped = queues[i].pop(0)
                    queues[i].append(x_to_push)
                    x_to_push = x_popped
                    X.append(x_popped)
                X = list(reversed(X))
                X_next, X_skip = parent(X, Z, conv=False)
                return X_next, X_skip

        return CausalConvCell()

    def make_queue(self, B, xt, Z=None):
        rate = self.params['rate']
        k, _, c_out = self.params['kernel']
        c_in = xt.get_shape()[1].value

        self.queues = queues = [
            tf.FIFOQueue(
                rate, dtypes=xt.dtype,
                name='queue_%d_%s' % (i, self.name)
            )
            for i in range(k - 1)
        ]

        flush_ops = [
            tf.cond(q.size() > 0, q.dequeue, lambda: tf.constant(float('nan')))
            for _ in range(rate) for q in queues
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
        X = list(reversed(X))

        X_next, X_skip = self(X, Z, conv=False)

        ops = struct(
            fill=fill_ops, push=push_ops, flush=flush_ops,
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
                XX = self._do_conv(X, 'x_%s' % self.name)
            else:
                wx = self.get_weight('kernel_x_%s')
                XX = tf.einsum('btlu,tluv->bv', X, wx)
            XX = tf.nn.relu(XX)
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
