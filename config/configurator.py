# @Time   : January, 2023
# @Author : Dr. Yong Zheng
# @Email  : yzheng66@iit.edu

"""
mopo.config.configurator
################################
"""

import re
import yaml

from utils.argument_list import *
from utils import set_color


class Config(object):
    """
        Configurator module that load the defined parameters.
        It loads internal configurations (i.e., default.yaml) first, then loads configurations from external files.
        The external configurations can overwrite the configurations in internal configurations.
    """

    def __init__(self, config_file_list=None):
        """ Class initialization.
            Args:
                * config_file_list (list of str): the external config file
        """
        self._init_parameters_category()
        self.yaml_loader = self._build_yaml_loader()
        self.file_config_dict = self._load_config_files(config_file_list)
        self._merge_external_config_dict()

        self.internal_config_dict = self._load_config_files(['yaml/system.yaml', 'yaml/scalarization.yaml', 'yaml/moea.yaml'])
        self.final_config_dict = self._get_final_config_dict()

    def _init_parameters_category(self):
        """ Load parameters into different parameter categories. """
        self.parameters = dict()
        self.parameters['Dataset'] = dataset_arguments
        self.parameters['User'] = user__arguments
        self.parameters['Outputs'] = outputs_arguments
        self.parameters['General'] = general_arguments
        self.parameters['Constraints'] = constraints_arguments
        self.parameters['Scalarization'] = scalarization_arguments
        self.parameters['MOEA'] = moea_arguments

    def _build_yaml_loader(self):
        """ Build YAML loader by using regular expression """
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(
                u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X
            ), list(u'-+0123456789.')
        )
        return loader

    def _convert_config_dict(self, config_dict):
        """ Convert the str parameters to their original type.
            Args:
                * config_dict (dict): dict of configurations
        """
        for key in config_dict:
            param = config_dict[key]
            if not isinstance(param, str):
                continue
            try:
                value = eval(param)
                if value is not None and not isinstance(value, (str, int, float, list, tuple, dict, bool)):
                    value = param
            except (NameError, SyntaxError, TypeError):
                if isinstance(param, str):
                    if param.lower() == "true":
                        value = True
                    elif param.lower() == "false":
                        value = False
                    else:
                        value = param
                else:
                    value = param
            config_dict[key] = value
        return config_dict

    def _load_config_files(self, file_list):
        """ Load a list of configuration files
            Args:
                * file_list: a list of YMAL files
        """
        file_config_dict = dict()
        if file_list:
            for file in file_list:
                with open(file, 'r', encoding='utf-8') as f:
                    file_config_dict.update(yaml.load(f.read(), Loader=self.yaml_loader))
        return file_config_dict

    def _merge_external_config_dict(self):
        """ Merge external configurations """
        external_config_dict = dict()
        external_config_dict.update(self.file_config_dict)
        self.external_config_dict = external_config_dict

    def _update_internal_config_dict(self, file):
        """ Update configurations from internal/default YAML file
            Args:
                * file: a single YAML file
        """
        with open(file, 'r', encoding='utf-8') as f:
            config_dict = yaml.load(f.read(), Loader=self.yaml_loader)
            if config_dict is not None:
                self.internal_config_dict.update(config_dict)
        return config_dict

    def _get_final_config_dict(self):
        """ Get final configurations by overwriting configurations in internal files from external files """
        final_config_dict = dict()
        final_config_dict.update(self.internal_config_dict)
        final_config_dict.update(self.external_config_dict)
        return final_config_dict

    def __setitem__(self, key, value):
        """ Set a value in configurations by using a string key """
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config_dict[key] = value

    def __getattr__(self, item):
        """ Validate attributes in configuration dict """
        if 'final_config_dict' not in self.__dict__:
            raise AttributeError(f"'Config' object has no attribute 'final_config_dict'")
        if item in self.final_config_dict:
            return self.final_config_dict[item]
        raise AttributeError(f"'Config' object has no attribute '{item}'")

    def __getitem__(self, item):
        """ Update configurations from internal/default YAML file """
        if item in self.final_config_dict:
            return self.final_config_dict[item]
        else:
            return None

    def __contains__(self, key):
        """ Validate a key in configuration dict """
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.final_config_dict

    def __str__(self):
        """ Return configurations to be printed out in console """
        args_info = '\n'
        for category in self.parameters:
            args_info += set_color(category + ' Hyper Parameters:\n', 'pink')
            args_info += '\n'.join([(set_color("{}", 'cyan') + " =" + set_color(" {}", 'yellow')).format(arg, value)
                                    for arg, value in self.final_config_dict.items()
                                    if arg in self.parameters[category]])
            args_info += '\n\n'

        args_info += set_color('Other Hyper Parameters: \n', 'pink')
        args_info += '\n'.join([
            (set_color("{}", 'cyan') + " = " + set_color("{}", 'yellow')).format(arg, value)
            for arg, value in self.final_config_dict.items()
            if arg not in {
                _ for args in self.parameters.values() for _ in args
            }.union({'config'})
        ])
        args_info += '\n\n'
        return args_info
