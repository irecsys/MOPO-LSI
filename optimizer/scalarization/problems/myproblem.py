# @Time   : January, 2023
# @Author : Dr. Yong Zheng
# @Email  : yzheng66@iit.edu

"""
mopo.optimizer.scalarization.problems.MyProblem
################################################
"""

from optimizer.scalarization.problems.scalarproblem import ScalarProblem
from optimizer.utils_optimizer import *


class MyProblem(ScalarProblem):
    """ Example of a Scalar problem defined by your own。
        In this example, we optimize PosESG, TE only。
    """

    def set_objectives(self):
        """ Add objectives into the list."""

        exp = get_objective_esg(self.data, self.w, self.data.pos_esg_dims,
                                self.data.client_preferences_esg_dims) \
              - get_objective_tracking_err(self.data, self.w)

        if not exp.is_dcp():
            raise ValueError("Problem is not disciplined.")
        else:
            self.exp = exp
        self.objectives = cvx.Maximize(self.exp)
        self.data.logger.info('Optimizer: objectives have been added.')
