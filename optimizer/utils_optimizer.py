# @Time   : January, 2023
# @Author : Dr. Yong Zheng
# @Email  : yzheng66@iit.edu

"""
mopo.optimizer.utils_optimizer
################################################
"""

import numpy as np
import importlib
import cvxpy as cvx

from data_loader.dataset import Dataset
from pymoo.decomposition.asf import ASF
from pymoo.mcdm.pseudo_weights import PseudoWeights


def get_problem_moea(problem_name):
    """ Automatically select problem class based on problem name
        Args:
            * problem_name (str): problem name
        Returns:
            * MOEA problem class
    """

    problem_submodule = [
        'moea.problems'
    ]

    problem_file_name = problem_name.lower()
    problem_module = None
    for submodule in problem_submodule:
        module_path = '.'.join(['optimizer', submodule, problem_file_name])
        if importlib.util.find_spec(module_path, __name__):
            problem_module = importlib.import_module(module_path, __name__)
            break

    if problem_module is None:
        raise ValueError('`problem_name` [{}] is not the name of an existing MOEA problem.'.format(problem_name))
    problem_class = getattr(problem_module, problem_name)
    return problem_class


def get_problem_scalar(problem_name):
    """ Automatically select problem class based on problem name
        Args:
            * problem_name (str): problem name
        Returns:
            * Scarlarization problem class
    """

    problem_submodule = [
        'scalarization.problems'
    ]

    problem_file_name = problem_name.lower()
    problem_module = None
    for submodule in problem_submodule:
        module_path = '.'.join(['optimizer', submodule, problem_file_name])
        if importlib.util.find_spec(module_path, __name__):
            problem_module = importlib.import_module(module_path, __name__)
            break

    if problem_module is None:
        raise ValueError('`problem_name` [{}] is not the name of an existing MOEA problem.'.format(problem_name))
    problem_class = getattr(problem_module, problem_name)
    return problem_class


def get_esg_score(data: Dataset, w, esg_dim):
    """ Return ESG score on an ESG dimension. The score is weighted score, where weights are fund allocations
        Args:
            * data: the Dataset object
            * w: list of fund allocations
            * esg_dim (string): an ESG dimension
        Returns:
            ESG score on a single ESG dimension
    """
    return w @ data.funds_esg_norm[esg_dim].values


def get_esg_score_dict(data: Dataset, w, list_esg_dim):
    """ Return ESG scores in a dict, key = dim, value = score
        It is used in analyzer only.
        Args:
            * data: the Dataset object
            * w: list of fund allocations
            * list_esg_dim: a list of ESG dimensions
        Returns:
            ESG score in a dict, key = dim, value = score
    """
    dict_scores = {}
    for dim in list_esg_dim:
        dict_scores[dim] = (w * data.funds[dim]).sum()
    return dict_scores


def get_objective_tracking_err(data: Dataset, w, moea=False):
    """ Return the TE objective (squared errors).
        Args:
            * data: the Dataset object
            * w: list of fund allocations
            * moea: client preference is multiplied with TE, if it is scalarization method or moea is False
        Returns:
            the objective of TE
    """
    if moea is False:
        return data.client_preferences_esg_groups['market'][0] * cvx.quad_form((w - data.b), data.cov_matrix)
    else:
        return (((w - data.b).reshape(1, data.num_funds_total)
                 @ data.cov_matrix.values @ (w - data.b)))


def get_objective_esg(data: Dataset, w, list_esg_dims: list, client_preferences_esg_dims: dict):
    """ Return the weighted average of the ESG score on ESG dimensions.
        Weights are client preferences.
        client_preferences_esg_dims is None, if it is run by a process of MOEA optimization
        Args:
            * data: the Dataset object
            * w: list of fund allocations
            * list_esg_dims (list): a list of ESG dimensions
            * client_preferences_esg_dims (dict): key = dim, value = [preference, strength]
        Returns:
            weighted average of ESG score from a list of ESG dimensions
    """
    score_total = 0
    count = 0
    for dim in list_esg_dims:
        dim_score = get_esg_score(data, w, dim)
        if client_preferences_esg_dims is None:
            # produces average score in MOEA
            score_total += dim_score
            count += 1.0
        else:
            # produce weighted average score in scalarization
            dim_preference = client_preferences_esg_dims[dim][0]
            score_total += dim_preference * dim_score
            count += dim_preference
    return score_total / count


def get_constraint_w_exclusion(data: Dataset, w):
    """ Return exclusion constraint to exclude the baseline funds.
        Args:
            * data: the Dataset object
            * w: list of fund allocations
        Returns:
            exclusion constraint to exclude the baseline funds
    """
    return data.funds['exclude'].values @ w


def get_constraint_tracking_err(data: Dataset, w, moea=False):
    """ Return constraint to limit ex-ante tracking error.
        Args:
            * data: the Dataset object
            * w: list of fund allocations
            * moea: used for MOEA methods or not
        Returns:
            constraint to limit ex-ante tracking error
    """
    adj_ratio = data.config['adj_ratio']
    if moea is False:  # scalarization methods require a quad form
        TE = cvx.quad_form((w - data.b), data.cov_matrix)
    else:  # MOEAs require a value
        TE = get_objective_tracking_err(data, w, True)
    return TE - (data.TE_cap / adj_ratio) ** 2


def get_constraint_w_individual_max(data: Dataset, w, moea=False):
    """ Return the constraints of max weights for individual funds.
        Args:
            * data: the Dataset object
            * w: list of fund allocations
            * moea: indicate whether it is used for MOEAs
        Returns:
            constraints of max weights for individual funds
    """
    max_w = np.minimum(data.config['weight_init_up_lim'], data.funds['MaxW'].values)
    if moea is False:
        return w - max_w
    else:
        return [w - max_w]


def get_constraints_asset_ratio_leq(data: Dataset, w):
    """ Return the asset allocation constraint with leq relationship
        Args:
            * data: the Dataset object
            * w: list of fund allocations
        Returns:
            constraints of asset allocations with leq relationship
    """
    constraints_asset_ratio = []
    for asset in data.bmk_asset_alloc['AssetClass']:
        dev = data.config['dev_asset_alloc'][asset]
        alloc_bmk = data.get_bmk_asset_alloc(data.client_option, asset)
        constraints_asset_ratio += get_context_leq_constraints({asset: (alloc_bmk + dev)}, data.funds_asset_alloc, w)
    return constraints_asset_ratio


def get_constraints_asset_ratio_geq(data: Dataset, w):
    """ Get asset allocation constraint with geq relationship
        Args:
            * data: the Dataset object
            * w: list of fund allocations
        Returns:
            constraints of asset allocations with geq relationship
    """
    constraints_asset_ratio = []
    for asset in data.bmk_asset_alloc['AssetClass']:
        dev = data.config['dev_asset_alloc'][asset]
        alloc_bmk = data.get_bmk_asset_alloc(data.client_option, asset)
        constraints_asset_ratio += get_context_geq_constraints({asset: (alloc_bmk - dev)}, data.funds_asset_alloc, w)
    return constraints_asset_ratio


def get_constraints_esg_geq(data: Dataset, w, list_pos_esg, moea=False):
    """ Get ESG constraints: PosESG in sol >= PosESG in bmk.
        Args:
            * data: the Dataset object
            * w: list of fund allocations
            * list_pos_esg: list of PosESG dimensions
            * moea: used for MOEA methods or not
        Returns:
            ESG constraints on PosESG dimensions
    """
    constraints_esg_geq = []
    for esg_dim in list_pos_esg:
        if moea is False:
            # only add the ones with strength == high
            strength = data.client_preferences_esg_dims[esg_dim][1]
            if strength == "high":
                pos_bmk = (data.lineup * data.funds_esg_norm[esg_dim]).sum()
                constraints_esg_geq += get_context_geq_constraints({esg_dim: pos_bmk}, data.funds_esg_norm, w)
        else:
            # add all in MOEAs
            pos_bmk = (data.lineup * data.funds_esg_norm[esg_dim]).sum()
            constraints_esg_geq += get_context_geq_constraints({esg_dim: pos_bmk}, data.funds_esg_norm, w)
    return constraints_esg_geq


def get_constraints_esg_leq(data: Dataset, w, list_neg_esg, moea=False):
    """ Get ESG constraints: NegESG in sol <= NegESG in bmk.
        Args:
            * data: the Dataset object
            * w: list of fund allocations
            * list_neg_esg: list of NegESG dimensions
            * moea: used for MOEA methods or not
        Returns:
            ESG constraints on NegESG dimensions
    """
    constraints_esg_leq = []
    for esg_dim in list_neg_esg:
        if moea is False:
            strength = data.client_preferences_esg_dims[esg_dim][1]
            if strength == "high":
                neg_bmk = (data.lineup * data.funds_esg_norm[esg_dim]).sum()
                constraints_esg_leq += get_context_leq_constraints({esg_dim: neg_bmk}, data.funds_esg_norm, w)
        else:
            neg_bmk = (data.lineup * data.funds_esg_norm[esg_dim]).sum()
            constraints_esg_leq += get_context_leq_constraints({esg_dim: neg_bmk}, data.funds_esg_norm, w)
    return constraints_esg_leq


def get_context_geq_constraints(context_constraints, matrix, w):
    """ Get constraints <= context-specific.
        Args:
            * context_constraints (dict): columns from p to be constrained
            along with values. >= is the sign used in all cases. Example:
            p['CarbonRiskScore'] >= 9.5. To adapt to MOEA, we use constraints <= context-specific
            * matrix: the objective from which we can retrieve values
            * w: list of fund allocations
    """
    constraints = []
    for key, value in context_constraints.items():
        constraints.append(matrix[key].values @ w - value)
    return constraints


def get_context_leq_constraints(context_constraints, matrix, w):
    """ Get context-specific <= constraints.
        Args:
            * context_constraints (dict): columns from p to be constrained
            along with values. <= is the sign used in all cases. Example:
            p['pfv'] <= 0.85.
            * matrix: the objective from which we can retrieve values
            * w: list of fund allocations
    """
    constraints = []
    for key, value in context_constraints.items():
        constraints.append(matrix[key].values @ w - value)
    return constraints


def return_tracking_err(data: Dataset, model_name):
    """ Calculate ex-ante tracking error of final portfolio.
        It is used in the result reports.
        Args:
            * data: the Dataset object
            * model_name: it is used to locate the solution of fund allocations
            * adj_ratio: adjustment ratio for TE
        Returns:
            ex-ante tracking error
    """
    cov = data.cov_matrix
    weight_col = model_name
    adj_ratio = data.config['adj_ratio']
    TE_exante = (
            (data.funds[weight_col].values - data.b).reshape(
                1, len(data.funds)
            )
            @ cov.values
            @ (data.funds[weight_col].values - data.b)
    )
    return pow(TE_exante[0], 0.5) * adj_ratio


def get_optimal_solution(nF, weights, method):
    """ Return the index of selected optimal solution
        Args:
            * nF (ndarrary): normalized objective space
            * weights: client preferences for the purpose of selections
            * method: either ASF or PW method
        Returns:
            the index of selected optimal solution
    """
    weights = np.array(weights)
    if method == "ASF":
        decomp = ASF()
        i = decomp.do(nF, 1 / weights).argmin()
        return i
    elif method == "PW":
        j = PseudoWeights(weights).do(nF)
        return j
