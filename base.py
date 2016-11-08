from __future__ import division
import os.path as osp
import numpy as np
import tensorflow as tf


def make_session(frac):
    """ Create a tf.Session(), limiting fraction of gpu that is allocated. """
    return tf.Session(config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=frac)
    ))


def make_placeholders(variables, dtype='tf.float32'):
    """
    Input: a dict{name: shape} or {name: (shape, dtype_str)}
           <shape> is a tuple of ints
           <dtype_str> is a str like 'tf.float32', 'tf.int32', etc

    If `dtype_str` is not given for a placeholder,
    it will use the one passed into this function,
    which is tf.float32 by default.

    Usage:
         variables = {
           'X_pl': (4, 5),
           'Y_pl': ((2, 3), 'tf.int32')
         }
         for var in make_placeholders(variables):
             exec(var)
    """
    commands = []
    for name, args in variables.items():
        if len(args) == 2 and isinstance(args[1], basestring):
            shape, dtype = args
        else:
            shape = args
        commands.append(
            "{0} = tf.placeholder({2}, {1}, name='{0}')".format(
                name, shape, dtype
            )
        )
    return commands


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


class struct(dict):

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


def _default_value(params_dict, key, value):
    if key not in params_dict:
        params_dict[key] = value


def _validate_axes(axes):
    if isinstance(axes, int):
        axes = [axes]
    assert axes is None or isinstance(axes, list)
    return axes
