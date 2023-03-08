# @Time   : January, 2023
# @Author : Dr. Yong Zheng
# @Email  : yzheng66@iit.edu

"""
mopo.optimizer.moea.problems.MOEAProblem
################################################
"""

import numpy as np

from data_loader.dataset import Dataset
from config import Config
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from optimizer.utils_optimizer import *


class SumWeightRepair(Repair):
    """ Create a repair operator as a constraint to make sure sum of weights = 1
        These repair operators are required to represent equality constraints in ElementwiseProblem
    """

    def __init__(self, dataset: Dataset):
        self.data = dataset
        super().__init__()

    def _do(self, problem, pop, **kwargs):
        for k in range(len(pop)):
            x = pop[k]

            while sum(x) > 1 + 1e-6:
                item = np.random.choice(self.data.num_funds_total)
                if sum(x) - x[item] >= 1:
                    x[item] = 0
                else:
                    x[item] -= sum(x) - 1

            while sum(x) < 1 - 1e-6:
                item = np.random.choice(self.data.num_funds_total)
                if self.data.funds['MaxW'].values[item] - x[item] <= 1 - sum(x):
                    x[item] = self.data.funds['MaxW'].values[item]
                else:
                    x[item] += 1 - sum(x)
        return pop


class MOEAProblem(ElementwiseProblem):
    """ Define a MOEA Problem as an ElementwiseProblem,
        including objectives and constraints
    """

    def __init__(self, dataset: Dataset, config: Config, **kwargs):
        """ Class initialization
            Args:
                * config (Config): configurations loaded from YAML file
                * dataset (Dataset): loaded Dataset
        """
        num_esg = len(config['client_pos_esg']) + len(config['client_neg_esg'])
        super().__init__(n_var=dataset.num_funds_total,  # len of w
                         n_obj=3 + num_esg,  # number of objectives: POS, NEG, TE, Selected ESG dimensions
                         n_constr=(1 + 2 * dataset.num_funds_total + 2 * dataset.num_asset_class + num_esg),
                         # num of constraints
                         xl=np.array([0 for _ in range(dataset.num_funds_total)]),  # lower bound of w
                         xu=np.array(dataset.funds['MaxW'].values))  # upper bound of w

        self.data = dataset
        self.config = config

    def _evaluate(self, x, out, *args, **kwargs):
        """ Define objectives and constraints
            Non-dominated relations will be evaluated in each running iteration

            Args:
                * x: fund allocations to be learned
                * out: pymoo dict which stores objectives and constraints
            Returns:
                None
        """
        # add objectives
        objectives = self.set_objectives(x)
        out["F"] = np.column_stack(objectives)

        # add constraints
        constraints = self.set_constraints(x)
        out["G"] = np.column_stack(constraints)

    def set_constraints(self, x):
        """ Define constraints
            Args:
                * x: fund allocations to be learned
            Returns:
                * constraints: a list of constraints
        """
        constraints = []
        # add constraints: non-negative weights
        constraints.append([-1*x])

        # add constraints: tracking err
        constraints.append(get_constraint_tracking_err(self.data, x, True))

        # add constraints: max weights for individual funds
        constraints.append(get_constraint_w_individual_max(self.data, x, True))

        # add constraints: asset allocation constraint, number = 2 * self.data.numAssetClass
        constraints_asset_ratio_leq = get_constraints_asset_ratio_leq(self.data, x)
        constraints_asset_ratio_geq = get_constraints_asset_ratio_geq(self.data, x)
        [constraints.append(leq) for leq in constraints_asset_ratio_leq]
        [constraints.append(-1 * geq) for geq in constraints_asset_ratio_geq]

        # add constraints: ESG scores in solution must be better than the ones in benchmark
        constraints_esg_leq = get_constraints_esg_leq(self.data, x, self.config['client_neg_esg'], True)
        constraints_esg_geq = get_constraints_esg_geq(self.data, x, self.config['client_pos_esg'], True)
        [constraints.append(leq) for leq in constraints_esg_leq]
        [constraints.append(-1 * geq) for geq in constraints_esg_geq]

        return constraints

    def set_objectives(self, x):
        """ Define objectives

            Args:
                * x: fund allocations to be learned
            Returns:
                * objectives: a list of objectives
        """
        objectives = []
        # add objective: tracking err
        tracking_err = get_objective_tracking_err(self.data, x, True)
        objectives.append(tracking_err)
        # tracking_err_constraint = get_constraint_tracking_err(self.data, x, True)
        # objectives.append(tracking_err_constraint)

        # add objective: overall PosESG
        pos_esg = -1*get_objective_esg(self.data, x, self.data.pos_esg_dims.keys(), None)  # negative values for pos_esg
        objectives.append(pos_esg)

        # add objective: overall NegESG
        neg_esg = get_objective_esg(self.data, x, self.data.neg_esg_dims.keys(), None)
        objectives.append(neg_esg)

        # add objective: list of PosESG dims selected by clients
        for dim in self.config['client_pos_esg']:
            objectives.append(-1 * get_esg_score(self.data, x, dim))
        # add objective: list of NegESG dims selected by clients
        for dim in self.config['client_neg_esg']:
            objectives.append(get_esg_score(self.data, x, dim))

        return objectives
