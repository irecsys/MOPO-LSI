# @Time   : January, 2023
# @Author : Dr. Yong Zheng
# @Email  : yzheng66@iit.edu

"""
mopo.optimizer.moea.problems.MyProblem
################################################
"""
import numpy as np

from data_loader.dataset import Dataset
from config import Config
from pymoo.core.problem import ElementwiseProblem
from optimizer.moea.problems.moeaproblem import MOEAProblem
from optimizer.utils_optimizer import *


class MyProblem(MOEAProblem):
    """ Example of a MOEA problem defined by your own。
        In this example, we optimize PosESG, NegESG, TE only。
    """
    def __init__(self, dataset: Dataset, config: Config, **kwargs):
        """ Class initialization
            Args:
                * config (Config): configurations loaded from YAML file
                * dataset (Dataset): loaded Dataset
        """
        super().__init__(dataset, config, **kwargs)
        # ElementwiseProblem.__init__(self, n_var=dataset.num_funds_total,  # len of w
        #                  n_obj=3,  # number of objectives: POS, NEG, TE, Selected ESG dimensions
        #                  n_constr=(1 + dataset.num_funds_total + 2 * dataset.num_asset_class),
        #                  # num of constraints
        #                  xl=np.array([0 for _ in range(dataset.num_funds_total)]),  # lower bound of w
        #                  xu=np.array(dataset.funds['MaxW'].values))  # upper bound of w
        #
        # self.data = dataset
        # self.config = config

    def set_constraints(self, x):
        """ Get constraints to be set by overwriting the method in parent class
            Args:
                * x: fund allocations to be learned
            Returns:
                * constraints: a list of constraints
        """
        constraints = []
        # add constraints: tracking err
        constraints.append(get_constraint_tracking_err(self.data, x, True))

        # add constraints: max weights for individual funds
        constraints.append(get_constraint_w_individual_max(self.data, x, True))

        # add constraints: asset allocation constraint, number = 2 * self.data.numAssetClass
        constraints_asset_ratio_leq = get_constraints_asset_ratio_leq(self.data, x)
        constraints_asset_ratio_geq = get_constraints_asset_ratio_geq(self.data, x)
        [constraints.append(leq) for leq in constraints_asset_ratio_leq]
        [constraints.append(-1 * geq) for geq in constraints_asset_ratio_geq]

        return constraints

    def set_objectives(self, x):
        """ Get objectives to be set by overwriting the method in parent class
            Args:
                * x: fund allocations to be learned
            Returns:
                * objectives: a list of objectives
        """
        objectives = []
        # add objective: tracking err
        tracking_err = get_objective_tracking_err(self.data, x, True)
        objectives.append(tracking_err)

        # add objective: overall PosESG
        pos_esg = -get_objective_esg(self.data, x, self.data.pos_esg_dims, None)  # negative values for pos_esg
        objectives.append(pos_esg)

        # add objective: overall NegESG
        neg_esg = get_objective_esg(self.data, x, self.data.neg_esg_dims, None)
        objectives.append(neg_esg)

        return objectives
