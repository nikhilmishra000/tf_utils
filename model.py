from __future__ import division
import cPickle as pickle
import numpy as np
import tensorflow as tf

from base import struct


class Model(object):

    def __init__(self, opts):
        self.opts = opts
        self.funcs = struct()

    @property
    def params(self):
        return self.session.graph.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES)

    @property
    def param_shapes(self):
        return [tuple(map(int, var.get_shape())) for var in self.params]

    @property
    def num_params(self):
        return sum([np.prod(shape) for shape in self.param_shapes])

    def save(self, path):
        self.saver.save(self.session, '{0}.ckpt'.format(path))
        pickle.dump(self.opts, open('{0}.opts'.format(path), 'wb'))
        print 'saved model to {0}'.format(path)

    def restore(self, path):
        opts = pickle.load(open('{0}.opts'.format(path), 'rb'))
        print opts
        self.saver.restore(self.session, '{0}.ckpt'.format(path))
        print 'loaded model from {0}'.format(path)
        return opts

    def make_function(self, name, inputs, outputs):
        self.funcs[name] = struct(inputs=inputs, outputs=outputs)

    def make_train_op(self, loss, scope):
        """
        Make a training op that minimizes the given loss.
        opts.solver_type: {"RMSProp", "Adam"}
        opts.alpha: learning rate
        opts.beta1, opts.beta2: parameters, (momentum/gamma for RMSP)
        """
        epsilon, opts = 1e-6, self.opts
        with tf.variable_scope(scope):
            SolverType = eval("tf.train.{0}Optimizer".
                              format(opts["solver_type"]))
            step = tf.Variable(0, name="global_step", trainable=False)
            if 'epsilon' in opts:
                epsilon = opts["epsilon"]

            solver = SolverType(opts["alpha"], opts["beta1"],
                                opts["beta2"], epsilon=epsilon)
            if 'grad_clip' in opts:
                gclip = opts['grad_clip']
                grads = solver.compute_gradients(loss)
                grads = [(tf.clip_by_value(grad, -gclip, +gclip), var)
                         for grad, var in grads]
                train_op = solver.apply_gradients(grads, global_step=step)
            else:
                train_op = solver.minimize(loss, global_step=step)
            return step, train_op
