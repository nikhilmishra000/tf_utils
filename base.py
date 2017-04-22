from __future__ import division
import os.path as osp
import numpy as np
import tensorflow as tf


def make_session(frac=None):
    """
    Create a tf.Session(), limiting fraction of gpu that is allocated.
    """
    if frac is None:
        return tf.Session()

    return tf.Session(config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=frac)
    ))


def make_placeholders(variables):
    """
    Returns a dict{name: placeholder}.

    :param variables: dict{name: shape} or {name: (shape, dtype)}
    where `shape` is a list/tuple of ints
    and `dtype` is is Tensorflow dtype (defaults to `tf.float32`)
    """
    placeholders = {}
    for name, args in variables.items():
        if len(args) == 2 and isinstance(args[1], basestring):
            shape, dtype = args
        else:
            shape, dtype = args, tf.float32

        placeholders[name] = tf.placeholder(dtype, shape, name=name)

    return placeholders


def scoped_variable(var_name, scope_name, **kwargs):
    """
    Get a variable from a scope, or create it if it doesn't exist.
    **kwargs will be passed to tf.get_variable if a new one is created.

    :param var_name: the variable name
    :param scope_name: the scope name
    """
    try:
        with tf.variable_scope(scope_name) as scope:
            return tf.get_variable(var_name, **kwargs)
    except ValueError:
        with tf.variable_scope(scope_name, reuse=True) as scope:
            return tf.get_variable(var_name, **kwargs)


def make_scoped_cell(CellType, **scope_kwargs):
    """
    Take a cell from `tf.nn.rnn_cell`,
    and make a version of it that sets `reuse` in its scope as needed.

    For example:
    ```
    from tf.nn.rnn_cell import BasicLSTMCell
    ScopedLSTMCell = tfu.make_scoped_cell(BasicLSTMCell)
    ```

    Now, `ScopedLSTMCell` can be used in place of `BasicLSTMCell`,
    and it should take care of reusing correctly.
    """
    class ScopedCell(CellType):

        def __init__(self, scope_name, *args, **kwargs):
            self.name = scope_name
            super(ScopedCell, self).__init__(*args, **kwargs)

        def __call__(self, X, H):
            try:
                with tf.variable_scope(self.name, **scope_kwargs) as scope:
                    return super(ScopedCell, self).__call__(X, H)
            except ValueError:
                with tf.variable_scope(self.name, reuse=True, **scope_kwargs) as scope:
                    return super(ScopedCell, self).__call__(X, H)

    ScopedCell.__name__ = "Scoped%s" % CellType.__name__
    return ScopedCell


class struct(dict):
    """
    A dict that exposes its entries as attributes.
    """

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


def structify(obj):
    """
    Modify `obj` by replacing `dict`s with `tfu.struct`s.
    """
    if isinstance(obj, dict):
        obj = type(obj)(**{
            key: structify(val) for key, val in obj.items()
        })
    elif isinstance(obj, list):
        obj = [structify(val) for val in obj]
    return obj

"""
Below here should not exposed.
"""


def _default_value(params_dict, key, value):
    if key not in params_dict:
        params_dict[key] = value


def _validate_axes(axes):
    if isinstance(axes, int):
        axes = [axes]
    assert axes is None or isinstance(axes, list)
    return axes


def _get_existing_vars(names_and_scopes):
    variables = {}
    for name, scope in names_and_scopes:
        try:
            variables[(name, scope)] = scoped_variable(name, scope)
        except ValueError:
            pass
    return variables
