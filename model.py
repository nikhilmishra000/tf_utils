from __future__ import division
import dill as pickle
import numpy as np
import tensorflow as tf

from base import struct, structify


class Function(object):

    def __init__(self, inputs, outputs, session=None, name='function'):
        """
        Create a function interface to tf.Session.run().
        Note that when calling the function, you must pass inputs with kwargs.

        Usage:
        >>> func = Function(inputs, outputs)
        >>> output1, output2 = func(input1=val1, input2=val2)

        @param inputs: a dict or struct of {str: placeholder}
        @param outputs: anything can be evaluated by tf.Session.run()
        @param session: if not given, use `tf.get_default_session()`.
        @param name: of the function
        """
        in_str = Function.generate_string(inputs)
        out_str = Function.generate_string(outputs)

        self.__doc__ = """( % s) = %s(%s)""" % (out_str, name, in_str)

        if session is None:
            session = tf.get_default_session()
        assert session is not None,  \
            "No Session was given, and there is no default Session."
        self.session = session
        self.inputs, self.outputs = inputs, outputs

    def __call__(self, **kwargs):
        feed = {pl: kwargs[name]
                for name, pl in self.inputs.items()}
        result = self.session.run(self.outputs, feed_dict=feed)
        result = structify(result)
        return result

    def __str__(self):
        return '< tfu.Function: %s >' % self.__doc__

    def __repr__(self):
        return str(self)

    @staticmethod
    def generate_string(arg):
        if isinstance(arg, tf.Tensor) \
           or isinstance(arg, tf.Operation) \
           or isinstance(arg, tf.Variable):
            return arg.name
        elif isinstance(arg, list):
            return ', '.join(Function.generate_string(a) for a in arg)
        elif isinstance(arg, dict):
            return Function.generate_string(arg.values())


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
      self.session = ...

      # make some placeholders
      X_pl, Y_pl = ...

      # build the graph
      prediction = ...
      loss = ...

      global_step, learning_rate, train_op = self.make_train_op(loss)

      self.functions = [
        ('fit', {'X': X_pl, 'Y': Y_pl},
         [train_op, loss]),
        ('score', {'X': X_pl, 'Y': Y_pl}, [loss, prediction]),
        ('predict', {'X': X_pl}, [prediction])
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
    scope_name = ''

    def __init__(self, opts):
        """
        :param opts: a dict that contains all information needed
                     to build the graph (layer sizes, solver params, etc)
        """
        struct.__init__(self)
        self['opts'] = opts
        self.functions = []

    @property
    def params(self):
        """
        The model's trainable parameters, as a list(tf.Variable).
        """
        return [v for v in self.session.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                if v.name.startswith(self.scope_name)]

    @property
    def param_shapes(self):
        """
        A list(tuple) of the shapes of `self.params`.
        """
        return [var.get_shape().as_list() for var in self.params]

    @property
    def param_names(self):
        """
        A list(str) of the names of `self.params`.
        """
        return [var.name for var in self.params]

    @property
    def num_params(self):
        """
        The total number of trainable parameters in this model.
        """
        return sum([np.prod(shape) for shape in self.param_shapes])

    @property
    def variables(self):
        return [v for v in self.session.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                if v.name.startswith(self.scope_name)]

    def save(self, path, **kwargs):
        """
        Save this model, pickling its parameter values, opts struct, and kwargs.
        @param path: str, where to save it
        """
        data = dict(**kwargs)
        for tensor in self.variables:
            data[tensor.name] = tensor.eval(session=self.session)
        data.update(self.opts)

        if 'session' in data:
            data.pop('session')

        pickle.dump(data, open(path, 'wb'))
        print 'saved model to', path

    @classmethod
    def restore(cls, path, **kwargs):
        """
        Restore a model by:
        * Un-pickle-ing a dict from `path`.
        * Recreating the Model object from that dict.
        * Restoring its parameter values.

        @param path: str
        @param kwargs: override saved values
        """
        opts = structify(pickle.load(open(path, 'rb')))
        for key, val in kwargs.items():
            opts[key] = val
        self = cls(opts).load_from(opts)
        print 'loaded %s params from: %s' % (type(self).__name__, path)
        return self

    def load_from(self, path_or_dict):
        """
        Restore saved model parameters,
        given either a dict of {str: np.array}
        or the path to a pickled version of such a dict.
        """
        if isinstance(path_or_dict, basestring):
            opts = pickle.load(open(path_or_dict, 'rb'))
        else:
            opts = path_or_dict
        for v in self.variables:
            self.session.run(v.assign(opts.pop(v.name)))
        return self

    def finalize(self, init_list=None):
        """
        Should be called at the end of the `__init__` method.

        Does the following:
        * Create instancemethods using `tfu.Model.make_function()`.
        * Initializes variables (either specified by `init_list`, or all uninitialized variables).
        """
        for name, inputs, outputs in self.functions:
            self.make_function(name, inputs, outputs)

        if init_list is None:
            needs_init = set(
                self.session.run(tf.report_uninitialized_variables())
            )
            init_list = [v for v in tf.global_variables()
                         if v.name.partition(':')[0] in needs_init]
            self.session.run(tf.variables_initializer(init_list))

        elif isinstance(init_list, list):
            self.session.run(tf.initialize_variables(init_list))

    def make_function(self, name, inputs, outputs):
        """
        Create an interface like an instancemethod.

        See `tfu.Function` for details.
        """
        self[name] = Function(inputs, outputs, self.session, name)

    def make_train_op(self, loss, var_list=None):
        """
        Make a training op that minimizes `loss` w.r.t `var_list`.
        Groups any `tf.GraphKeys.UPDATE_OPS` to be run with the train op.
        Returns the iteration number, learning rate, and train_op.

        Required in `self.opts`:
        solver_type: {"RMSProp", "Adam"}
        alpha: learning rate
        beta1, beta2: parameters, (momentum / gamma for RMSP)

        Optional:
        var_list: will be passed to Optimizer.minimize
        epsilon: constant for numerical stability (default 1e-6)
        lr_decay, lr_step: learning rate multiplies by `lr_decay` every `lr_step` iterations.
        min_alpha: learning rate stops decaying once it gets to `min_alpha`

        @TODO add gradient clipping and support for other optimizition algos
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

        solver = SolverType(
            alpha, opts["beta1"], opts["beta2"], epsilon=epsilon
        )
        if 'grad_clip' not in opts:
            train_op = solver.minimize(loss, step, var_list)
        else:
            clip = opts['grad_clip']
            grads = solver.compute_gradients(loss, var_list)
            clipped = [
                (tf.clip_by_value(grad, -clip, +clip), var)
                for grad, var in grads
                if grad is not None
            ]
            train_op = solver.apply_gradients(clipped, step)

        update_ops = tf.get_collection(
            tf.GraphKeys.UPDATE_OPS, scope=self.scope_name
        )
        ops = [train_op] + update_ops
        return step, alpha, tf.group(*ops)

    def __repr__(self):
        return object.__repr__(self)

    def __str__(self):
        return object.__str__(self)
