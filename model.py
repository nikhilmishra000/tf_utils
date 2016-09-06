from __future__ import division
import cPickle as pickle
import numpy as np
import tensorflow as tf

from base import struct


class Model(struct):

    def __init__(self, opts):
        struct.__init__(self)
        self['opts'] = opts

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

    def save(self, path, **kwargs):
        for key, val in kwargs.items():
            self.opts[key] = val
        self.saver.save(self.session, '{0}.ckpt'.format(path))
        pickle.dump(self.opts, open('{0}.opts'.format(path), 'wb'))
        print 'saved model to {0}'.format(path)

    @classmethod
    def restore(cls, path):
        opts = pickle.load(open('{0}.opts'.format(path), 'rb'))
        self = cls(opts)
        self.saver.restore(self.session, '{0}.ckpt'.format(path))
        print 'loaded model from {0}'.format(path)
        return self

    def finalize(self):
        for name, inputs, outputs in self.functions:
            self.make_function(name, inputs, outputs)

        self.saver = tf.train.Saver()
        self.session.run(tf.initialize_all_variables())

    def make_function(self, name, inputs, outputs):
        """
        Create an instancemethod interface to tf.Session.run().

        Usage:
        >>> self.make_function('func_name',
                               {'arg1': arg1_pl, 'arg2': arg2_pl},
                               [res1_tensor, res2_tensor])
        >>> res1_val, res2_val = self.func_name(arg1=arg1_val, arg2=arg2_val)

        Note that when calling the function, you must pass inputs with kwargs.
        """

        def function(**values):
            feed = {pl: values[name]
                    for name, pl in inputs.items()}
            result = self.session.run(outputs, feed_dict=feed)
            if len(result) == 1:
                result = result[0]
            return result

        in_str = ', '.join(inputs)
        out_str = ', '.join(o.name for o in outputs)

        function.__doc__ = """
        (%s) = %s(%s)
        """ % (out_str, name, in_str)

        self[name] = function

    def make_train_op(self, loss):
        """
        Make a training op that minimizes the given loss.
        Returns the iteration number, learning rate, and train_op.

        Required:
        opts.solver_type: {"RMSProp", "Adam"}
        opts.alpha: learning rate
        opts.beta1, opts.beta2: parameters, (momentum / gamma for RMSP)

        Optional:
        opts.epsilon: constant for numerical stability(default 1e-6)
        opts.lr_decay, opts.lr_step: learning rate multiplies by `lr_decay` every `lr_step` iterations.
        opts.min_alpha: learning rate stops decaying once it gets to `min_alpha`
        opts.grad_clip: clips gradients to be within[-grad_clip, +grad_clip]
        """
        opts = self.opts

        SolverType = eval("tf.train.{0}Optimizer".
                          format(opts["solver_type"]))
        step = tf.Variable(0, name="global_step", trainable=False)
        epsilon = opts.get('epsilon', 1e-6)

        if 'lr_step' in opts and 'lr_decay' in opts:
            alpha = tf.train.exponential_decay(
                opts['alpha'], step,
                opts['lr_step'], opts['lr_decay']
            )

            alpha = tf.maximum(alpha, opts.get('min_alpha', 0),
                               name='learning_rate')

        else:
            alpha = tf.Variable(opts['alpha'],
                                name='learning_rate', trainable=False)

        solver = SolverType(alpha, opts["beta1"],
                            opts["beta2"], epsilon=epsilon)
        if 'grad_clip' in opts:
            gclip = opts['grad_clip']
            grads = solver.compute_gradients(loss)
            grads = [(tf.clip_by_value(grad, -gclip, +gclip), var)
                     for grad, var in grads]
            train_op = solver.apply_gradients(grads, global_step=step)
        else:
            train_op = solver.minimize(loss, global_step=step)

        return step, alpha, train_op
