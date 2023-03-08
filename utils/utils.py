# @Time   : January, 2023
# @Author : Dr. Yong Zheng
# @Email  : yzheng66@iit.edu

"""
mopo.utils.utils
################################################
"""

import datetime
import importlib
import os
import random
import numpy as np
import pickle


def init_seed(seed):
    """ init random seed for random functions in numpy, etc
        Args:
            seed (int): random seed
    """
    random.seed(seed)
    np.random.seed(seed)


def get_local_time():
    """ Get current time
        Returns:
            str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur


def ensure_dir(dir_path):
    """ Make sure the directory exists, if it does not exist, create it
        Args:
            dir_path (str): directory path
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_model(model_name):
    """ Automatically select model class based on model name
        Args:
            model_name (str): model name
        Returns:
            Optimizer: model class
    """
    model_submodule = [
        'scalarization', 'moea'
    ]

    model_file_name = model_name.lower()
    model_module = None
    for submodule in model_submodule:
        module_path = '.'.join(['optimizer', submodule, model_file_name])
        if importlib.util.find_spec(module_path, __name__):
            model_module = importlib.import_module(module_path, __name__)
            break

    if model_module is None:
        raise ValueError('`model_name` [{}] is not the name of an existing model.'.format(model_name))
    model_class = getattr(model_module, model_name)
    return model_class


def get_model_type(model_name):
    """ Automatically select model class based on model name
        Args:
            model_name (str): model name
        Returns:
            model_type (str)
    """
    model_submodule = [
        'scalarization', 'moea'
    ]

    model_file_name = model_name.lower()
    model_type = None
    for submodule in model_submodule:
        module_path = '.'.join(['optimizer', submodule, model_file_name])
        if importlib.util.find_spec(module_path, __name__):
            model_type = submodule
            break

    if model_type is None:
        raise ValueError('`model_name` [{}] is not the name of an existing model.'.format(model_name))

    return model_type


def normalize(x, old_min, old_max):
    """ min-max normalization
        Args:
            * x: values to be normalized
            * old_min: min in old scale
            * old_max: max in old scale
        Returns:
            * normalized values
    """
    x_norm = (x - old_min) / (old_max - old_min)
    return x_norm


def save_object(obj, filename):
    """ save object serialization into external file
        Args:
            * obj: the object to be serialized
            * filename: external filename
        Returns:
            * None
    """
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp)


def load_object(filename):
    """ load object serialization into Python object
        Args:
            * filename: external file which saves object serialization
        Returns:
            * object
    """
    obj = None
    with open(filename, 'rb') as inp:  # read external file
        obj = pickle.load(inp)
    return obj
