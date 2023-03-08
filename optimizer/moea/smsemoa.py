# @Time   : January, 2023
# @Author : Dr. Yong Zheng
# @Email  : yzheng66@iit.edu

"""
mopo.optimizer.moea.SMSEMOA
################################################
"""

from data_loader.dataset import Dataset
from config import Config
from optimizer.moea.moea import MOEA
from optimizer.moea.problems.moeaproblem import SumWeightRepair
from pymoo.algorithms.moo import sms


class SMSEMOA(MOEA):
    """ SMSEMOA: a MOEA algorithms as solver
    """

    def __init__(self, dataset: Dataset, config: Config):
        """ Class initialization
            Args:
                * config (Config): configurations loaded from YAML file
                * dataset (Dataset): loaded Dataset
        """
        super().__init__(dataset, config)
        self.model_name = 'SMSEMOA'

    def get_algorithm(self):
        """ Return an instance of SMSEMOA in pymoo library
        """
        alg = sms.SMSEMOA(pop_size=self.config['n_population'],
                          n_offsprings=self.config['n_offsprings'],
                          repair=SumWeightRepair(self.data),
                          eliminate_duplicates=True)
        return alg
