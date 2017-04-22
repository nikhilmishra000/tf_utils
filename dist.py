from base import struct
import numpy as np
import tensorflow as tf
from cached_property import cached_property


class NormalDist(struct):

    I = 'identity'

    @staticmethod
    def create(mu, sig=None, logsig=None):
        self = NormalDist(mu=mu)
        if sig is None and logsig is not None:
            self.sig, self.logsig = tf.exp(logsig), logsig
        elif logsig is None and sig is not None:
            self.sig, self.logsig = sig, tf.log(1e-12 + sig)
        else:
            raise ValueError("Give exactly one of {sig, logsig}")

        self.eps = tf.placeholder_with_default(
            tf.truncated_normal(tf.shape(mu)),
            mu.get_shape().as_list()
        )
        self.sample = s = mu + tf.stop_gradient(self.eps) * self.sig
        s.set_shape(mu.shape)
        return self

    def kld_from(self, other, axis=None):
        if axis is None:
            axis = self.mu.get_shape().ndims - 1
        if not isinstance(axis, list):
            axis = [axis]

        if other is NormalDist.I:
            kld = 0.5 * tf.square(self.mu) + \
                0.5 * tf.square(self.sig) - self.logsig - 0.5

        else:
            assert isinstance(other, NormalDist)
            kld = (self.logsig - other.logsig) + \
                  (
                tf.square(old.sig) + tf.square(other.mu - self.mu)
            ) / (2 * tf.square(self.sig))

        return tf.reduce_sum(kld, axis)
