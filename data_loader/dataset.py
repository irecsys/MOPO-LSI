# @Time   : January, 2023
# @Author : Dr. Yong Zheng
# @Email  : yzheng66@iit.edu

"""
mopo.dataloader.dataset
################################################
"""

import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from config import Config
from logging import getLogger
from utils.utils import get_model_type


class Dataset(object):
    """ Dataset class which stores funds, covariance matrix, benchmark, lineup vectors, constraint settings, etc."""

    def __init__(self, config: Config) -> None:
        """ Class initialization
            Args:
                * config (Config): configurations loaded from YAML file
        """
        self.config = config
        self.config["path_inputs"] = 'data/' + config['dataset'] + '/'
        self.logger = getLogger()

        # List of class attributes: individual variables
        self.client_option = None  # client option of risk levels: conservative, moderate, aggressive
        self.TE_cap = 0  # the Cap for tracking error
        self.esg_norm_up_lim = -1  # up bound for ESG normalizations
        self.esg_norm_low_lim = 0  # low bound for ESG normalizations
        self.num_funds = 0  # number of candidate funds
        self.num_funds_excluded = 0  # number of excluded funds
        self.num_funds_total = 0  # number of all funds
        self.num_asset_class = 0  # number of asset classes

        # List of class attributes: ndarray or list
        self.lineup = None  # lineup vector for selected benchmark; vector w in doc
        self.client_preferences_objectives = []  # preference vector on objectives for selecting a single solution in MOEA
        self.b = []  # benchmark weights; vector b in doc
        self.w = []  # solution weights; vector w in doc

        # List of class attributes: dict
        self.client_preferences_esg_groups = {}  # client preferences on ESG groups from user inputs, e.g., climate_change: [ 0.23, 'high' ]
        self.client_preferences_esg_dims = {}  # client preferences on ESG dims from user inputs; vector p in doc, e.g., thermal_coal: [ 0.23, 'high' ]
        self.pos_esg_dims = {}  # a dict for positive esg dim: esg group, e.g., climate_action: climate_change
        self.neg_esg_dims = {}  # a dict for negative esg dim: esg group, e.g., thermal_coal: climate_change

        # List of class attributes: dataframe
        self.cov_matrix = None  # covariance matrix; MatV in doc
        self.bmk_asset_alloc = None  # benchmark asset allocations; MatA in doc
        self.funds_asset_alloc = None  # fund distributions over 6 asset classes; MatD in doc
        self.funds_esg_norm = None  # normalized ESG scores for the list of funds; MatE in doc
        self.funds = None  # the dataframe/whole data

    def load_data(self):
        """ Load inputs data, including funds, cov matrix, benchmark allocations
        """

        # load the list of funds ######################################################
        funds = pd.read_csv(self.config["path_inputs"] + self.config['data_funds'])
        self.num_funds = (~funds.exclude).sum()
        self.num_funds_excluded = funds.shape[0] - self.num_funds
        self.num_funds_total = funds.shape[0]
        self.logger.info('Loaded: data \'' + self.config['dataset'] + '\' has ' + f'{str(self.num_funds)} funds')

        # load covariance matrix ######################################################
        cov_matrix = pd.read_csv(self.config["path_inputs"] + self.config['data_cov_matrix'])
        cov_matrix.set_index('secid', inplace=True)
        cov_matrix = cov_matrix.loc[funds.secid.values, funds.secid.values]
        # validation of funds and covariance data
        assert (
                funds.secid.values == cov_matrix.index.values
        ).all(), "cov matrix indexes not aligned!"
        self.logger.info('Loaded: covariance matrix')
        self.cov_matrix = cov_matrix

        # load benchmark asset allocations #############################################
        bmk_asset_alloc = pd.read_csv(self.config["path_inputs"] + self.config['data_bmk'])
        bmk_asset_alloc["AssetClass"] = bmk_asset_alloc.AssetClass.apply(lambda x: x.lower())
        self.client_option = self.config['client_option']
        # validate values in asset allocation are ratio data or not
        if bmk_asset_alloc[self.client_option][bmk_asset_alloc[self.client_option] > 1].count() > 0:
            for option in self.config['options']:
                bmk_asset_alloc[option] /= bmk_asset_alloc[option].sum()
        self.bmk_asset_alloc = bmk_asset_alloc
        self.num_asset_class = bmk_asset_alloc.shape[0]

        # calculate lineup for benchmark
        cols = list(self.bmk_asset_alloc["AssetClass"].unique())
        real_funds = funds.loc[funds["exclude"] == False]
        rescaled_asset_exposures = real_funds[cols] / real_funds[cols].sum(axis=0)
        funds["lineup_vector"] = rescaled_asset_exposures @ self.bmk_asset_alloc[self.client_option].values
        funds["lineup_vector"].fillna(0, inplace=True)
        funds["lineup_vector"] = funds["lineup_vector"] / funds["lineup_vector"].sum(axis=0)
        self.lineup = funds["lineup_vector"].values
        self.logger.info('Loaded: benchmark information and lineup vectors')

        # fund distributions over 6 asset classes #############################################
        cols = list(self.bmk_asset_alloc["AssetClass"].unique())
        funds[cols] = funds[cols].fillna(0)
        funds[cols] = funds[cols].astype(float)
        funds[cols] = funds[cols].div(funds[cols].sum(axis=1), axis=0)
        cols.insert(0, 'secid')
        self.funds_asset_alloc = funds[cols]

        # load ESG preferences ######################################################
        self.pos_esg_dims = self.config['pos_esg_dims']
        self.neg_esg_dims = self.config['neg_esg_dims']
        self.client_preferences_esg_groups = self.config['client_preferences']

        # replicate user preferences on ESG groups into each ESG dimension
        # load it only for scalarization method using weighted sum
        model_type = get_model_type(self.config['model'])
        if model_type == 'scalarization':
            if self.client_preferences_esg_groups is not None:
                for dim_pos in self.pos_esg_dims.keys():
                    esg_group = self.pos_esg_dims[dim_pos]
                    self.client_preferences_esg_dims[dim_pos] = self.client_preferences_esg_groups[esg_group]
                for dim_neg in self.neg_esg_dims.keys():
                    esg_group = self.neg_esg_dims[dim_neg]
                    self.client_preferences_esg_dims[dim_neg] = self.client_preferences_esg_groups[esg_group]
                self.logger.info('Loaded: client preferences for scalarization methods')
            else:
                self.logger.error('Client preferences on ESG groups are not configured.')
        else:
            # load preferences on objectives for the selectio of single optimal solution in MOEA
            self.client_preferences_objectives = self.config['client_preferences_moea']
            self.logger.info('Loaded: client preferences for MOEA methods')

        # load ESG scores for funds ######################################################
        list_esg_dims = list(self.pos_esg_dims.keys()) + list(self.neg_esg_dims.keys())
        self.funds_esg_norm = funds[list_esg_dims]
        self.funds_esg_norm.insert(0, 'secid', funds['secid'])

        self.funds = funds

        # using min-max normalization to scale it to [1e-100, TE2]
        self.TE_cap = self.config['TE_cap']
        self.esg_norm_low_lim = self.config['esg_norm_low_lim']
        self.esg_norm_up_lim = self.config['esg_norm_up_lim']
        self.esg_norm_up_lim = self.TE_cap if self.esg_norm_up_lim <= 0 else self.esg_norm_up_lim
        df_funds_esg = funds.loc[funds['exclude'] == False, list_esg_dims]
        df_bmk_esg = funds.loc[funds['exclude'] == True, list_esg_dims]
        df_funds_esg = pd.DataFrame(self.normalization(df_funds_esg, self.esg_norm_low_lim, self.esg_norm_up_lim), columns=list_esg_dims)
        df_funds_esg = df_funds_esg.fillna(0)  # fill NaN by using 0
        self.funds_esg_norm = pd.concat([df_funds_esg, df_bmk_esg])
        self.logger.info('Loaded: normalized ESG scores for the fund list')

        # create vector b ################################################################
        bmk = self.bmk_asset_alloc[self.client_option]
        self.b = [0] * self.num_funds + bmk.tolist()

        # calculate bounds of weights for individual funds ###############################
        self.calculate_individual_weight_min_max_bounds()

    def normalization(self, df_esg, low_lim, up_lim):
        """ normalize ESG dataframe to make sure ESG values are in same scale of sum squared of TE
            Args:
                * df_esg: a dataframe where rows refer to funds, and cols refer to ESG dims
                * low_lim: low bound of ESG score
                * up_lim: up bound of the ESG core
            Returns:
                normalized ESG ndarray
        """
        low = low_lim
        up = up_lim ** 2
        scaler = MinMaxScaler(feature_range=(low, up))
        df_esg = scaler.fit_transform(df_esg)
        return df_esg

    def get_bmk_asset_alloc(self, option, asset_class):
        """ get the allocation in benchmark by given an investment optinal and asset class
            Args:
                * option (string): investment option, e.g., conservative, moderate, aggressive
                * asset_class (string): one of the asset classes
            Returns:
                the allocation ratio
        """
        return self.bmk_asset_alloc[option][self.bmk_asset_alloc['AssetClass'] == asset_class]

    def calculate_individual_weight_min_max_bounds(self) -> None:
        """ Calculate upper/lower bounds of weight of funds constituting benchmark."""

        max_equity_allo, max_fixed_income_allo = self.calculate_individual_max_weight()

        self.funds["MaxW"] = self.config['weight_init_up_lim']
        self.funds["MinW"] = self.config['weight_init_low_lim']
        self.funds.loc[self.funds.isEquity == 1, "MaxW"] = max_equity_allo
        self.funds.loc[self.funds.isFixedIncome == 1, "MaxW"] = max_fixed_income_allo
        self.funds.loc[self.funds.exclude == True, "MaxW"] = 0

    def calculate_individual_max_weight(self):
        """ Calculate maximal fund allocation for equity and fixed incomes, respectively
            Returns:
                * max_equity_allo: maximal fund allocation for equity
                * max_fixed_income_allo: maximal fund allocations for fixed incomes
        """
        temp = self.bmk_asset_alloc[[self.client_option, 'AssetClass']]
        temp["bmk_stock_flag"] = temp.AssetClass.apply(
            lambda x: 1 if "stock" in x else 0
        )
        target_equity_alloc = temp.loc[temp.bmk_stock_flag == 1, self.client_option].sum()
        target_fixed_income_alloc = temp.loc[
            temp.bmk_stock_flag == 0, self.client_option
        ].sum()

        cols_stock = [
            col for col in list(temp["AssetClass"].unique()) if "stock" in col
        ]
        cols_bond = [
            col for col in list(temp["AssetClass"].unique()) if "stock" not in col
        ]
        self.funds["isEquity"] = self.funds[cols_stock].sum(axis=1) > 0.9
        self.funds["isFixedIncome"] = self.funds[cols_bond].sum(axis=1) > 0.9

        pos_esg_dims = self.config['pos_esg_dims']

        feasibility_condition = (
                self.funds[pos_esg_dims].notnull().apply(any, axis=1) & self.funds.exclude == 0
        )
        num_feasible_equity = self.funds.loc[feasibility_condition, "isEquity"].sum()
        num_feasible_fixed_income = self.funds.loc[
            feasibility_condition, "isFixedIncome"
        ].sum()

        max_equity_allo = min(1, max(
            self.config['weight_init_up_lim'],
            (target_equity_alloc / num_feasible_equity * 1.5)) if num_feasible_equity > 0 else 1)

        max_fixed_income_allo = min(1, max(
            self.config['weight_init_up_lim'],
            (target_fixed_income_alloc / num_feasible_fixed_income * 1.5)) if num_feasible_fixed_income > 0 else 1)

        return max_equity_allo, max_fixed_income_allo
