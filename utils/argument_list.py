# @Time   : January, 2023
# @Author : Dr. Yong Zheng
# @Email  : yzheng66@iit.edu

"""
mopo.utils.argument_list
################################################
"""


dataset_arguments = [
    'dataset', 'data_funds', 'data_cov_matrix', 'data_bmk'
]

user__arguments = [
    'client_option', 'client_preferences', 'client_pos_esg', 'client_neg_esg', 'client_preferences_moea',
    'preloaded_moea_results'
]

outputs_arguments = [
    'path_outputs',
    'output_analysis', 'output_hypervolume',
    'output_visualizations', 'output_visualize_dims',
    'output_funds_top_n'
]

general_arguments = [
    'seed', 'options', 'model',
    'pos_esg_dims', 'neg_esg_dims', 'esg_norm_up_lim', 'esg_norm_low_lim'
]

constraints_arguments = [
    'weight_init_up_lim', 'weight_init_low_lim', 'weight_final_low_lim',
    'TE_cap', 'adj_ratio',
    'dev_asset_alloc'
]

scalarization_arguments = [
    'objectives_pos_neg', 'ecos_eps', 'ecos_max_iter', 'ecos_verbose',
    'scalar_problem_name'
]

moea_arguments = [
    'client_preferences_moea', 'moea_verbose',
    'moea_problem_name', 'n_threads', 'n_population', 'n_offsprings', 'n_generations',
    'output_hypervolume_visualizations', 'optimal_selection'
]



