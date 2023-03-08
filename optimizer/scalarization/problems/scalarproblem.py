# @Time   : January, 2023
# @Author : Dr. Yong Zheng
# @Email  : yzheng66@iit.edu

"""
mopo.optimizer.scalarization.problems.ScalarProblem
################################################
"""

from data_loader.dataset import Dataset
from config import Config
from optimizer.utils_optimizer import *


class ScalarProblem(object):
    """ Define a Problem to be solved by scalarization methods,
        including objectives and constraints
    """
    def __init__(self, dataset: Dataset, config: Config):
        """ Class initialization
            Args:
                * config (Config): configurations loaded from YAML file
                * dataset (Dataset): loaded Dataset
        """
        self.data = dataset
        self.config = config

        self.w = cvx.Variable(shape=dataset.funds.shape[0])  # decision variable; the fund allocations to be learnt
        self.objectives = []  # list of objectives
        self.constraints = [cvx.sum(self.w) == 1]  # list of constraints
        self.exp = None  # expression of learning objective, i.e., weighted sum of multiple objectives

    def create_problem(self):
        """ Create and return a cvx.Problem by defining objectives and constraints
            Returns:
                * cvx.Problem
        """
        self.set_objectives()
        self.set_constraints()
        return cvx.Problem(self.objectives, self.constraints)

    def set_constraints(self):
        """ Add constraints into the list."""
        # add constraints: w >= 0
        self.constraints.extend([self.w >= 0])
        # add constraints: exclude baseline funds
        self.constraints.extend([get_constraint_w_exclusion(self.data, self.w) == 0])
        # add constraints: max weights for individual funds
        self.constraints.extend([get_constraint_w_individual_max(self.data, self.w) <= 0])
        # add constraints: tracking err
        self.constraints.extend([get_constraint_tracking_err(self.data, self.w) <= 0])
        # add constraints: asset allocation constraint
        constraints_asset_ratio_leq = get_constraints_asset_ratio_leq(self.data, self.w)
        constraints_asset_ratio_geq = get_constraints_asset_ratio_geq(self.data, self.w)
        [self.constraints.extend([leq <= 0]) for leq in constraints_asset_ratio_leq]
        [self.constraints.extend([geq >= 0]) for geq in constraints_asset_ratio_geq]
        # add constraints: ESG scores in solution must be better than the ones in benchmark
        constraints_esg_leq = get_constraints_esg_leq(self.data, self.w, self.data.neg_esg_dims)
        constraints_esg_geq = get_constraints_esg_geq(self.data, self.w, self.data.pos_esg_dims)
        [self.constraints.extend([leq <= 0]) for leq in constraints_esg_leq]
        [self.constraints.extend([geq >= 0]) for geq in constraints_esg_geq]
        self.data.logger.info('Optimizer: constraints have been added.')

    def set_objectives(self):
        """ Add objectives into the list."""

        # acquire True/False of two objectives: PosESG, NegESG
        pos_neg = self.config['objectives_pos_neg']
        pos_on = pos_neg[0]
        neg_on = pos_neg[1]

        if pos_on and neg_on:  # objectives: PosESG, NegESG, TE
            exp = get_objective_esg(self.data, self.w, self.data.pos_esg_dims.keys(), self.data.client_preferences_esg_dims) \
                  - get_objective_esg(self.data, self.w, self.data.neg_esg_dims.keys(), self.data.client_preferences_esg_dims) \
                  - get_objective_tracking_err(self.data, self.w)
        else:
            if pos_on:  # objectives: PosESG, TE
                exp = get_objective_esg(self.data, self.w, self.data.pos_esg_dims.keys(),
                                        self.data.client_preferences_esg_dims) \
                      - get_objective_tracking_err(self.data, self.w)
            if neg_on:  # objectives: NegESG, TE
                exp = 0 - get_objective_esg(self.data, self.w, self.data.neg_esg_dims.keys(),
                                            self.data.client_preferences_esg_dims) \
                      - get_objective_tracking_err(self.data, self.w)
            if not pos_on and not neg_on:  # objectives: TE
                exp = 0 - get_objective_tracking_err(self.data, self.w)

        if not exp.is_dcp():
            raise ValueError("Problem is not disciplined.")
        else:
            self.exp = exp
        self.objectives = cvx.Maximize(self.exp)
        self.data.logger.info('Optimizer: objectives have been added.')