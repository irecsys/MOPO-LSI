# @Time   : January, 2023
# @Author : Dr. Yong Zheng
# @Email  : yzheng66@iit.edu

"""
mopo.optimizer.scalarization.Weighted_Sum
################################################
"""

from data_loader.dataset import Dataset
from config import Config
from optimizer.utils_optimizer import *


class WeightedSum(object):
    """ Scalarization method: Weighted Sum """

    def __init__(self, dataset: Dataset, config: Config):
        """ Class initialization
            Args:
                * config (Config): configurations loaded from YAML file
                * dataset (Dataset): loaded Dataset
            Returns:
                None
        """

        self.model_type = 'scalarization'
        self.model_name = 'WeightedSum'

        self.data = dataset
        self.config = config

        self.problem_name = self.config['scalar_problem_name']  # the name of a scalarization problem
        self.problem = None  # the actual cvx.Problem to be solved
        self.w = None  # store the optimal fund allocations

    def set_problem(self):
        """ Set a problem to be solved. A problem defines objectives and constraints.
            YOu can define your own problems by extending ScalarProblem in mopo.optimizer.scalarization.problems.
        """
        problem = get_problem_scalar(self.problem_name)(self.data, self.config)
        self.problem = problem.create_problem()
        self.w = problem.w

    def run_optimization(self):
        """ Solve the scalarization problem by using ECOS as solver.
            Returns:
                * status: the status of the problem, e.g., optimal
                * funds: the list of funds for analyzer
                * TE: tracking error to be reported
        """

        eps = self.config['ecos_eps']
        max_iters = self.config['ecos_max_iter']
        verbose = self.config['ecos_verbose']

        self.problem.solve(
            solver=cvx.ECOS,
            warm_start=False,
            verbose=verbose,
            abstol=eps,
            reltol=eps,
            feastol=eps,
            max_iters=max_iters,
        )
        while (self.problem.status != "optimal") and (max_iters < 1e06):
            self.data.logger.info('ECOS failed to find a solution. Start trying more learning iterations...')
            max_iters = max_iters * 10
            self.problem.solve(
                solver=cvx.ECOS,
                warm_start=False,
                verbose=True,
                abstol=eps,
                reltol=eps,
                feastol=eps,
                max_iters=max_iters,
            )

        # create a new column by using model_name which stores the optimal solution, i.e., fund allocations
        self.data.funds[self.model_name] = self.w.value
        self.data.funds.loc[self.data.funds[self.model_name] < self.config['weight_final_low_lim'], self.model_name] = 0
        self.data.funds[self.model_name] = self.data.funds[self.model_name] / self.data.funds[self.model_name].sum()

        return self.problem.status, self.data.funds, return_tracking_err(self.data, self.model_name)




