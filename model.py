from __future__ import division
import dill as pickle
import numpy as np
import tensorflow as tf

from base import struct


class Model(struct):
    """
    A base class from which to derive different models.
    Provides methods and attributes for commonly-done things.
    Easy to save and restore a trained model.

    Most of the work is done in the __init__ method,
    which should look something like:

    ```
    def __init__(self, opts):
      super(Model, self).__init__(opts)
      self.session = tf.Session()

      # make some placeholders
      X, Y = ...

      # build the graph
      prediction = ...
      loss = ...

      global_step, learning_rate, train_op = self.make_train_op(loss)

      self.functions = [
        ('fit', {'X': X, 'Y': Y},
         [train_op, loss]),
        ('score', {'X': X, 'Y': Y}, [loss, prediction]),
        ('predict', {'X': X}, [prediction])
      ]

      self.finalize()
    ```

    Then, to train and query the model:

    ```
    for _ in range(num_iters):
      # get a batch of training data
      x, y = ...
      _, training_loss = self.fit(X=x, Y=y)

    # validation
    x_val, y_val = ...
    validation_loss, _ = self.score(X=x_val, Y=y_val)

    # make predictions
    x_test = ...
    predictions = self.predict(X=x_test)
    """

    def __init__(self, opts):
        """
        :param opts: a dict that contains all information needed
                     to build the graph (layer sizes, solver params, etc)
        """
        struct.__init__(self)
        self['opts'] = opts

    @property
    def params(self):
        """ The model parameters, as a list(tf.Variable) """
        return self.session.graph.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES)

    @property
    def param_shapes(self):
        """
        A list(tuple) of the shapes of model parameters.
        """
        return [var.get_shape().as_list() for var in self.params]

    @property
    def param_names(self):
        """
        A list(str) of the names of model parameters.
        """
        return [var.name for var in self.params]

    @property
    def num_params(self):
        """
        The total number of parameters in this model.
        """
        return sum([np.prod(shape) for shape in self.param_shapes])

    def save(self, path, **kwargs):
        """
        Take a snapshot of this model, saving the session with tf.train.Saver(),
        and pickle-ing the object's attributes, along with anything in kwargs.

        @param path: str, should not have a filename extension
        """
        for key, val in kwargs.items():
            self.opts[key] = val
        self.saver.save(self.session, '{0}.ckpt'.format(path))
        pickle.dump(self.opts, open('{0}.opts'.format(path), 'wb'))
        print 'saved model to {0}'.format(path)

    @classmethod
    def restore(cls, path):
        """
        Restore a model by:
        (1) Un-pickle-ing a dict from `"%s.opts" % path`
        (2) Recreating the Model object from that dict.
        (3) Restoring its session.

        Should be called as a classmethod of the derived class,
        not this base class.

        @param path: str, should not have a filename extension
        """
        opts = pickle.load(open('{0}.opts'.format(path), 'rb'))
        self = cls(opts)
        self.saver.restore(self.session, '{0}.ckpt'.format(path))
        print 'loaded model from {0}'.format(path)
        return self

    def finalize(self, init_list=True):
        """
        If `self.functions` is a list of tuples (str, dict, list),
        then this method will create instancemethods using `Model.make_function()`.
        Also gives this model a tf.train.Saver()
        and initializes all variables.

        If `init_list` is `True` (default), initializes all variables.
        Otherwise, it should be a list of variables.
        """
        for name, inputs, outputs in self.functions:
            self.make_function(name, inputs, outputs)

        if self.params:
            self.saver = tf.train.Saver()
        if init_list is True:
            self.session.run(tf.initialize_all_variables())
        elif isinstance(init_list, list):
            self.session.run(tf.initialize_variables(init_list))

    def make_function(self, name, inputs, outputs):
        """
        Create an instancemethod-like interface to tf.Session.run().

        Usage:
        >> > self.make_function('func_name',
                               {'arg1': arg1_placeholder, 'arg2': arg2_placeholder},
                               [res1_tensor, res2_tensor])
        >> > res1_val, res2_val = self.func_name(arg1=arg1_val, arg2=arg2_val)

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
        ( % s) = %s(%s)
        """ % (out_str, name, in_str)

        self[name] = function

    def make_train_op(self, loss, var_list=None):
        """
        Make a training op that minimizes the given loss.
        Returns the iteration number, learning rate, and train_op.

        All parameters should be in the dict that got passed to __init__

        Required:
        solver_type: {"RMSProp", "Adam"}
        alpha: learning rate
        beta1, beta2: parameters, (momentum / gamma for RMSP)

        Optional:
        var_list: will be passed to Optimizer.minimize
        epsilon: constant for numerical stability(default 1e-6)
        lr_decay, lr_step: learning rate multiplies by `lr_decay` every `lr_step` iterations.
        min_alpha: learning rate stops decaying once it gets to `min_alpha`
        grad_clip: clips gradients to be within[-grad_clip, +grad_clip]
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
            train_op = solver.minimize(loss, step, var_list)

        return step, alpha, train_op

    def __repr__(self):
        return object.__repr__(self)

    def __str__(self):
        return object.__str__(self)
