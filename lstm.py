from __future__ import division
import numpy as np
import tensorflow as tf
from python_ops import conv, deconv


class ConvLSTM(object):

    def __init__(self, param, tag):
        """ Convolutional LSTM cell:
            param['kernel'] = (kernel_width, kernel_height)
            param['input_dim'] = (width_in, height_in, channels_in)
            param['stride'] = (1, stride_w, stride_h, 1)
            param['output_channels'] = int
        """
        self._param, self._tag = param, str(tag)
        self.kernel_shape = param['kernel']

        W, H, C_in = param['input_dim']
        self.input_size = W * H * C_in
        self.input_shape = (W, H, C_in)
        self.input_stride = param['stride']

        _, stride_w, stride_h, _ = param['stride']
        W_out, H_out = int(np.ceil(W / stride_w)), int(np.ceil(H / stride_h))
        C_out = param['output_channels']
        self.output_size = W_out * H_out * C_out
        self.output_shape = (W_out, H_out, C_out)
        self.state_shape = (W_out, H_out, 2 * C_out)
        self.state_size = W_out * H_out * C_out * 2

    def __call__(self, X_in, S_prev):
        W_in, H_in, C_in = self.input_shape
        W_out, H_out, C_out = self.output_shape
        kW, kH = self.kernel_shape

        C_prev, H_prev = tf.split(3, 2, S_prev)

        init = tf.truncated_normal_initializer(stddev=0.1)
        with tf.variable_scope('conv_lstm' + self._tag, reuse=None,
                               initializer=init) as scope:

            X_ = conv(X_in, {'kernel': (kW, kH, C_in, 4 * C_out),
                             'stride': self.input_stride,
                             'pad': 'SAME'}, scope, 'conv_lstm_x')
            H_ = conv(H_prev, {'kernel': (kW, kH, C_out, 4 * C_out),
                               'stride':  (1, 1, 1, 1),
                               'pad': 'SAME'}, scope, 'conv_lstm_h')
            Xi, Xf, Xo, Xx = tf.split(3, 4, X_)
            Hi, Hf, Ho, Hx = tf.split(3, 4, H_)
            I, F, O, X = Xi + Hi, Xf + Hf, Xo + Ho, Xx + Hx

            C_next = tf.sigmoid(F) * C_prev + tf.sigmoid(I) * tf.tanh(X)
            H_next = tf.sigmoid(O) * tf.tanh(C_next)
            S_next = tf.concat(3, [C_next, H_next])

        return H_next, S_next


class DeconvLSTM(object):

    def __init__(self, param, tag):
        """ Convolutional LSTM cell:
            param['kernel'] = (kernel_width, kernel_height) 
            param['input_dim'] = (width_in, height_in, channels_in)
            param['stride'] = (1, stride_w, stride_h, 1)    
            param['output_dim'] = (width_out, height_out, channels_out)

            Make sure 'input_dim', 'stride' and 'output_dim' all agree.
            Otherwise it will pretend to work then complain
            at gradient calculation.
        """
        self._param, self._tag = param, str(tag)
        self.kernel_shape = param['kernel']

        W, H, C_in = param['input_dim']
        self.input_size = W * H * C_in
        self.input_shape = (W, H, C_in)
        self.input_stride = param['stride']

        W_out, H_out, C_out = param['output_dim']
        self.output_size = W_out * H_out * C_out
        self.output_shape = (W_out, H_out, C_out)
        self.state_shape = (W_out, H_out, 2 * C_out)
        self.state_size = W_out * H_out * C_out * 2

    def __call__(self, X_in, S_prev):
        B = int(X_in.get_shape()[0])
        W_in, H_in, C_in = self.input_shape
        W_out, H_out, C_out = self.output_shape
        kW, kH = self.kernel_shape

        C_prev, H_prev = tf.split(3, 2, S_prev)
        init = tf.truncated_normal_initializer(stddev=0.1)
        with tf.variable_scope('deconv_lstm' + self._tag, reuse=None,
                               initializer=init) as scope:

            X_ = deconv(X_in, {'kernel': (kW, kH, 4 * C_out, C_in),
                               'stride': self.input_stride,
                               'output': (B, W_out, H_out, 4 * C_out),
                               'pad': 'SAME'}, scope, 'deconv_lstm_x')
            H_ = deconv(H_prev, {'kernel': (kW, kH, 4 * C_out, C_out),
                                 'stride':  (1, 1, 1, 1),
                                 'output': (B, W_out, H_out, 4 * C_out),
                                 'pad': 'SAME'}, scope, 'deconv_lstm_h')
            Xi, Xf, Xo, Xx = tf.split(3, 4, X_)
            Hi, Hf, Ho, Hx = tf.split(3, 4, H_)
            I, F, O, X = Xi + Hi, Xf + Hf, Xo + Ho, Xx + Hx

            C_next = tf.sigmoid(F) * C_prev + tf.sigmoid(I) * tf.tanh(X)
            H_next = tf.sigmoid(O) * tf.tanh(C_next)
            S_next = tf.concat(3, [C_next, H_next])

        return H_next, S_next
