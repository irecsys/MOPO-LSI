# @Time   : January, 2023
# @Author : Dr. Yong Zheng
# @Email  : yzheng66@iit.edu

"""
mopo.utils.quick_start
################################################
"""

import warnings
import time
import os

from logging import getLogger
from utils.logger import init_logger
from utils.utils import init_seed, get_model, normalize, save_object, load_object
from optimizer.utils_optimizer import return_tracking_err, get_optimal_solution
from config import Config
from data_loader.dataset import Dataset
from utils.analyzer import *
from pymoo.indicators.hv import HV

warnings.filterwarnings("ignore")


def run_optimization(config_file_list=None):
    """ entry of the running. """

    # load configurations
    config = Config(config_file_list=config_file_list)
    init_seed(config['seed'])

    # logger initialization
    log_filepath = init_logger(config)
    logger = getLogger()
    filename = os.path.splitext(log_filepath)[0]

    logger.info(config)
    logger.info('MOPO-LSI version ' + str(config['version']))

    # load dataset
    timer_begins = time.time()
    dataset = Dataset(config)
    dataset.load_data()

    # run optimization
    model_name = config['model']

    optimizer = get_model(model_name)(dataset, config)
    optimizer.set_problem()

    if optimizer.model_type == 'scalarization':
        logger.info('Start running optimizations by using ' + model_name + '...')
        status, funds, TE = optimizer.run_optimization()

        timer_ends = time.time()
        running_time = round(timer_ends - timer_begins, 2)
        TE = round(TE, 5)
        if status == 'optimal':
            logger.log(70, 'Time cost: ' + str(running_time) + ' seconds. Optimization is done, with tracking error of ' + str(TE))
        else:
            logger.error('Time cost: ' + str(running_time) + ' seconds. Optimization is not achieved. See more details in error message.')

        # results analysis
        if config['output_analysis'] is True:
            logger.info('Start results analysis...\n')
            aly = Analyzer(dataset, config)
            aly.analyze_optimal_results()

        # save results to csv file
        output = filename + '.csv'
        funds.to_csv(output, header=True)
        logger.log(70, 'The optimal solution is saved to: ' + output)
        logger.log(70, 'The log information is saved to: ' + log_filepath)

    elif optimizer.model_type == 'moea':
        logger.info('Optimizer: objectives and constraints have been set.')

        results = None  # save results from MOEA
        if config['preloaded_moea_results'] is None or config['preloaded_moea_results'].lower() == 'none':
            # run optimizer and save results
            logger.info('Start running optimizations by using ' + model_name + '...')
            results = optimizer.run_optimization()
            save_object(results, filename+'.pkl')
            logger.log(70, 'The MOEA results are saved to: ' + filename + '.pkl')
        else:  # load results from external file
            logger.info('Loading MOEA results from ' + config['preloaded_moea_results'])
            results = load_object(config['preloaded_moea_results'])

        num_sols = results.F.shape[0]
        num_objs = results.F.shape[1]
        config['objective_names'] = ['market', 'pos_esg', 'neg_esg'] + config['client_pos_esg'] + config['client_neg_esg']
        bool_minimize = [True, False, True] + [False]*len(config['client_pos_esg']) + [True]*len(config['client_neg_esg'])
        client_preferences_moea = []
        [client_preferences_moea.append(config['client_preferences_moea'][obj]) for obj in config['client_preferences_moea'].keys()]
        client_preferences_moea = client_preferences_moea / np.sum(client_preferences_moea)

        timer_ends = time.time()
        running_time = round(timer_ends - timer_begins, 2)
        logger.log(70, 'Time cost: ' + str(running_time) + ' seconds. There are ' + str(num_sols) + ' non-dominated solutions by ' + model_name)

        # results analysis
        if config['output_analysis'] is True:
            logger.info('Start results analysis...\n')

            # get old scale in current result set
            objs = results.F
            actual_ideal = objs.min(axis=0)
            actual_nadir = objs.max(axis=0)

            # normalize objective space in the result set
            normalized_objs = normalize(objs, actual_ideal, actual_nadir)

            # get single optimal solution
            o = get_optimal_solution(normalized_objs, client_preferences_moea, config['optimal_selection'])
            # store fund allocations into data set
            dataset.funds[model_name] = results.X[o]
            dataset.funds.loc[dataset.funds[model_name] < config['weight_final_low_lim'], model_name] = 0
            dataset.funds[model_name] = dataset.funds[model_name] / dataset.funds[model_name].sum()
            TE = round(return_tracking_err(dataset, model_name), 5)
            logger.log(70, 'Selected optimal solution by ' + config['optimal_selection'] +
                       ' is achieved. Tracking error is ' + str(TE))

            # analyze the optimal results
            aly = Analyzer(dataset, config)
            aly.analyze_optimal_results(True)

            if config['output_visualizations']:
                # visualization: plot feasible solutions and optimal solution
                visualize_dims = config['output_visualize_dims']
                visualize_dims_index = []
                [visualize_dims_index.append(config['objective_names'].index(dim)) for dim in visualize_dims]
                if len(visualize_dims_index) == 2:
                    aly.plot_optimal_solution(objs, o, filename, visualize_dims, visualize_dims_index)
                else:
                    raise ValueError("You should define two dimensions for visualizations. " +
                                     "See 'output_visualize_dims' in moea.yaml.")

            # save the optimal solution to csv file
            output_solutions = filename + '_Optimal.csv'
            dataset.funds.to_csv(output_solutions, header=True)
            logger.log(70, 'The optimal solution is saved to: ' + output_solutions)

            # save all non-dominated solutions to csv file
            for i in range(0, num_sols):
                col= 'nd-'+str(i)
                dataset.funds[col] = results.X[i]
                dataset.funds.loc[dataset.funds[col] < config['weight_final_low_lim'], col] = 0
                dataset.funds[col] = dataset.funds[col] / dataset.funds[col].sum()
            output_solutions = filename + '_All.csv'
            dataset.funds.to_csv(output_solutions, header=True)
            logger.log(70, 'All non-dominated solutions are saved to: ' + output_solutions)

            if config['output_hypervolume_visualizations']:
                logger.info('Starting analysis of hypervolumes... It may take longer time...')
                aly.plot_hypervolumes(results, filename)

            # save logs
            logger.log(70, 'The log information is saved to: ' + log_filepath)
    else:
        logger.error('Failed to identify the configured optimization model. Please check your configurations.')


