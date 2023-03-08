# @Time   : January, 2023
# @Author : Dr. Yong Zheng
# @Email  : yzheng66@iit.edu

"""
mopo.utils.analyzer
################################
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from optimizer.utils_optimizer import get_esg_score_dict
from data_loader.dataset import Dataset
from config.configurator import Config
from logging import getLogger
from pymoo.core.result import Result
from pymoo.indicators.hv import Hypervolume
from utils.utils import normalize


class Analyzer(object):
    """ It's used to analyze the results by the optimizers. """

    def __init__(self, dataset: Dataset, config: Config) -> None:
        """ Class initialization
            Args:
                * data (Dataset): loaded data set
                * config (Config): configurations loaded from YAML file
            Returns:
                None
        """
        self.config = config
        self.data = dataset
        self.logger = getLogger()

    def analyze_optimal_results(self, moea=False):
        """ compare the optimal solution with baselines, output gains and top-N investments
            Args:
                * moea: if True, use equal weights in the calculation of gain metrics
        """
        # get allocation weights
        w_bmk = self.data.funds['lineup_vector']
        w_sol = self.data.funds[self.config['model']]

        # analysis on PosESG
        scores_pos_bmk = get_esg_score_dict(self.data, w_bmk, self.data.pos_esg_dims.keys())
        scores_pos_sol = get_esg_score_dict(self.data, w_sol, self.data.pos_esg_dims.keys())
        rst_pos = pd.DataFrame({'Solution': pd.Series(scores_pos_sol), 'Benchmark': pd.Series(scores_pos_bmk)})
        rst_pos['Ratio'] = (rst_pos['Solution'] - rst_pos['Benchmark']) / rst_pos['Benchmark']
        temp = []
        for dim in rst_pos.index.tolist():
            if moea is False: # use client preferences as weights for weighted average
                temp.append(self.data.client_preferences_esg_dims[dim][0])
            else:  # use equal weights in MOEA
                temp.append(1)
        rst_pos['Preference'] = temp
        gain_pos = f"{np.dot(rst_pos['Preference'], rst_pos['Ratio']).sum() / rst_pos['Preference'].sum():,.2%}"
        rst_pos['Ratio'] = pd.Series(["{0:.2f}%".format(val * 100) for val in rst_pos['Ratio']], index=rst_pos.index)
        self.logger.log(70, 'Gains on PosESG: ' + gain_pos)
        self.logger.info('Details on PosESG:\n{}\n'.format(rst_pos.to_string()))

        # analysis on NegESG
        scores_neg_bmk = get_esg_score_dict(self.data, w_bmk, self.data.neg_esg_dims.keys())
        scores_neg_sol = get_esg_score_dict(self.data, w_sol, self.data.neg_esg_dims.keys())
        rst_neg = pd.DataFrame({'Solution': pd.Series(scores_neg_sol), 'Benchmark': pd.Series(scores_neg_bmk)})
        rst_neg['Ratio'] = (rst_neg['Benchmark'] - rst_neg['Solution']) / rst_neg['Benchmark']
        temp = []
        for dim in rst_neg.index.tolist():
            if moea is False:  # use client preferences as weights for weighted average
                temp.append(self.data.client_preferences_esg_dims[dim][0])
            else:  # use equal weights in MOEA
                temp.append(1)
        rst_neg['Preference'] = temp
        gain_neg = f"{np.dot(rst_neg['Preference'], rst_neg['Ratio']).sum() / rst_neg['Preference'].sum():,.2%}"
        rst_neg['Ratio'] = pd.Series(["{0:.2f}%".format(val * 100) for val in rst_neg['Ratio']], index=rst_neg.index)
        self.logger.log(70, 'Gains on NegESG: ' + gain_neg)
        self.logger.info('Details on NegESG:\n{}\n'.format(rst_neg.to_string()))

        # print top-n investments
        invest = self.data.funds.sort_values(by=self.config['model'], ascending=False)
        cols = [self.config['model'], 'secid', 'name']
        top_n = self.config['output_funds_top_n']
        invest = invest[invest[self.config['model']] > 0][cols].head(top_n)
        self.logger.log(70, 'Top Investments (up to ' + str(top_n) + '): ')
        self.logger.info('\n{}\n'.format(invest.to_string()))

    def plot_optimal_solution(self, F, index, filename, visualize_dims, visualize_dims_index):
        """ Produce 2D visualizations, where the selected optimal solution is highlighted
            Args:
                * F: the objective space before normalization
                * index: the index of the optimal solution
                * filename: used to save png
                * visualize_dims: names of two selected dimensions
                * visualize_dims_index: indices of two selected dimensions in the objective space
        """
        viz_index_1 = visualize_dims_index[0]
        viz_index_2 = visualize_dims_index[1]
        viz_dim_1 = visualize_dims[0]
        viz_dim_2 = visualize_dims[1]

        plt.figure(figsize=(7, 5))
        plt.scatter(F[:, viz_index_1], F[:, viz_index_2], s=30, facecolors='none', edgecolors='blue', label="Solutions")
        plt.scatter(F[index, viz_index_1], F[index, viz_index_2], marker="x", color="red", s=200)
        plt.xlabel(viz_dim_1)
        plt.ylabel(viz_dim_2)
        plt.title("Objective Space")
        output_png = filename + '_sol.png'
        plt.savefig(output_png)
        self.logger.log(70, 'The visualization of optimal solution is saved to: ' + output_png)

    def plot_hypervolumes(self, res: Result, filename):
        """ Plot hypervolumes
            Args:
                * res: results set by MOEA
                * filename: used to save png
        """
        timer_begins = time.time()
        hist = res.history
        num_runs = len(hist)

        F = res.F
        num_objs = F.shape[1]
        approx_ideal = F.min(axis=0)
        approx_nadir = F.max(axis=0)

        nF = normalize(F, approx_ideal, approx_nadir)
        approx_ideal_norm = nF.min(axis=0)
        approx_nadir_norm = nF.max(axis=0)

        metric = Hypervolume(ref_point=[1.1]*num_objs,
                             norm_ref_point=False,
                             zero_to_one=True,
                             ideal=approx_ideal_norm,
                             nadir=approx_nadir_norm)
        objs_last = hist[num_runs - 1].opt.get('F')
        hv_last = round(metric.do(normalize(objs_last, approx_ideal, approx_nadir)), 2)

        timer_ends = time.time()
        running_time = round(timer_ends - timer_begins, 2)
        self.logger.log(70, 'Current hypervolume: ' + str(hv_last) + '. Time cost: ' + str(running_time) + ' seconds.')

        self.logger.info('Start plotting historical hypervolumes... It may take much longer time...')
        timer_begins = time.time()
        n_evals = []  # corresponding number of function evaluations
        hist_F = []  # the objective space values in each generation
        hist_cv = []  # constraint violation in each generation
        hist_cv_avg = []  # average constraint violation in the whole population

        for run in hist:
            # store the number of function evaluations
            n_evals.append(run.evaluator.n_eval)

            # retrieve the optimum from the algorithm
            opt = run.opt

            # store the least constraint violation and the average in each population
            hist_cv.append(opt.get("CV").min())
            hist_cv_avg.append(run.pop.get("CV").mean())

            # filter out only the feasible and append and objective space values
            feas = np.where(opt.get("feasible"))[0]
            objs = opt.get("F")[feas]
            hist_F.append(objs)

        metric = Hypervolume(ref_point=[1.1]*num_objs,
                             norm_ref_point=False,
                             zero_to_one=True,
                             ideal=approx_ideal_norm,
                             nadir=approx_nadir_norm)
        hv = [metric.do(normalize(_F, approx_ideal, approx_nadir)) for _F in hist_F]

        plt.figure(figsize=(7, 5))
        plt.plot(n_evals, hv, color='black', lw=0.7, label="Avg. CV of Pop")
        plt.scatter(n_evals, hv, facecolor="none", edgecolor='black', marker="p")
        plt.title("Convergence")
        plt.xlabel("Function Evaluations")
        plt.ylabel("Hypervolume")
        output_png = filename + '_hv.png'
        plt.savefig(output_png)

        timer_ends = time.time()
        running_time = round(timer_ends - timer_begins, 2)
        self.logger.info('Plotting is complete. Time cost: ' + str(running_time) + ' seconds.')
        self.logger.log(70, 'The plot of historical hypervolumes is saved to: ' + output_png)



