# @Time   : January, 2023
# @Author : Dr. Yong Zheng
# @Email  : yzheng66@iit.edu

"""
mopo.optimizer.moea.AGEMOEA2
################################################
"""

from data_loader.dataset import Dataset
from config import Config
from optimizer.moea.moea import MOEA
from optimizer.moea.problems.moeaproblem import SumWeightRepair
from pymoo.algorithms.moo import age2


class AGEMOEA2(MOEA):
    """ AGEMOEA2: a MOEA algorithms as solver
    """

    def __init__(self, dataset: Dataset, config: Config):
        """ Class initialization
            Args:
                * config (Config): configurations loaded from YAML file
                * dataset (Dataset): loaded Dataset
        """
        super().__init__(dataset, config)
        self.model_name = 'AGEMOEA2'

    def get_algorithm(self):
        """ Return an instance of AGEMOEA2 in pymoo library
        """
        alg = age2.AGEMOEA2(pop_size=self.config['n_population'],
                            n_offsprings=self.config['n_offsprings'],
                            repair=SumWeightRepair(self.data),
                            eliminate_duplicates=True)
        return alg
