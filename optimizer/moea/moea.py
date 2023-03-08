# @Time   : January, 2023
# @Author : Dr. Yong Zheng
# @Email  : yzheng66@iit.edu

"""
mopo.optimizer.moea.MOEA
################################################
"""

from abc import abstractmethod
from data_loader.dataset import Dataset
from config import Config
from optimizer.utils_optimizer import get_problem_moea

from multiprocessing.pool import ThreadPool
from pymoo.core.problem import StarmapParallelization
from pymoo.optimize import minimize


class MOEA(object):
    """ MOEA as a parent class """

    def __init__(self, dataset: Dataset, config: Config):
        """ Class initialization
            Args:
                * config (Config): configurations loaded from YAML file
                * dataset (Dataset): loaded Dataset
            Returns:
                None
        """
        self.model_type = 'moea'
        self.data = dataset
        self.config = config

        self.problem_name = self.config['moea_problem_name']
        self.problem = None

    def set_problem(self):
        """ Set a problem to be solved. A problem defines objectives and constraints.
            YOu can define your own problems by extending MOEAProblem in mopo.optimizer.moea.problems.
        """
        n_threads = self.config['n_threads']
        if n_threads <= 1:
            self.problem = get_problem_moea(self.problem_name)(self.data, self.config)
        else:
            pool = ThreadPool(n_threads)
            runner = StarmapParallelization(pool.starmap)
            self.problem = get_problem_moea(self.problem_name)(self.data, self.config, elementwise_runner=runner)

    @abstractmethod
    def get_algorithm(self):
        """ The subclass is a specific MOEA solver in pymoo library
            This function is required to be implemented to get an instance of the solver
        """
        return

    def run_optimization(self):
        """ Run MOEA optimization to seek non-dominated solutions by a specific MOEA solver
            Returns:
                * results: the result object by pymoo library which can be ustilized for analysis
                  see the list of attributes in this object, https://pymoo.org/interface/result.html
        """
        alg = self.get_algorithm()

        # the history is used for the purpose of visualizing historical hypervolumes only
        # it runs slowly, if it is set as True
        save_hist = False
        if self.config['output_hypervolume_visualizations'] is True:
            save_hist = True
        results = minimize(self.problem,  # MOEA problem to be solved
                           alg,  # a MOEA solver or algorithm
                           seed=self.config['seed'],
                           termination=('n_gen', self.config['n_generations']),  # terminate learning by MaxGen
                           verbose=self.config['moea_verbose'],  # output intermediate results or not
                           save_history=save_hist  # save all intermediate results or not
                           )
        return results
