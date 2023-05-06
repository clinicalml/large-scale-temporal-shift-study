import numpy as np
import pandas as pd
from os.path import dirname, abspath, join
from scipy.sparse import csr_matrix, csc_matrix
import sys
import os
import time
import joblib
import argparse
from datetime import datetime
import multiprocessing as mp
import multiprocessing_logging
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

from load_data_for_nonstationarity_scan import load_outcomes, load_covariates
from find_conditional_shift_examples import select_features_for_statsmodels

sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from logging_utils import set_up_logger
from model_utils import (
    train_logreg, 
    save_logreg, 
    train_statsmodels_logreg, 
    save_statsmodels_logreg, 
    get_predictions_from_statsmodels_logreg
)

def select_robust_features(coef_df_year1,
                           coef_df_year2,
                           feature_names,
                           logger):
    '''
    Select features with the same coefficient signs in both years
    @param coef_df_year1: pandas DataFrame, contains 'Feature' and 'Coefficient' columns, coefficients for year 1
    @param coef_df_year2: pandas DataFrame, contains 'Feature' and 'Coefficient' columns, coefficients for year 2
    @param feature_names: list of str, feature names
    @param logger: logger, for INFO messages
    @return: 1. list of int, indices of selected features
             2. list of str, names of selected features
    '''
    assert {'Feature', 'Coefficient'}.issubset(set(coef_df_year1.columns.values))
    assert {'Feature', 'Coefficient'}.issubset(set(coef_df_year2.columns.values))
    assert len(coef_df_year1) == len(feature_names)
    assert len(coef_df_year2) == len(feature_names)
    
    coef_df_merged = coef_df_year1.merge(coef_df_year2,
                                         on = 'Feature',
                                         suffixes = ['_year1', '_year2'])
    coef_df_merged['Coefficient_product'] = coef_df_merged['Coefficient_year1'].multiply(coef_df_merged['Coefficient_year2'])
    features_not_flipped = coef_df_merged.loc[coef_df_merged['Coefficient_product'] >= 0]['Feature'].values
    
    feature_name_to_idx       = {feature_names[idx]: idx for idx in range(len(feature_names))}
    feature_idxs_not_flipped  = [feature_name_to_idx[name] for name in features_not_flipped]
    feature_idxs_not_flipped.sort()
    feature_names_not_flipped = [feature_names[idx] for idx in feature_idxs_not_flipped]
    logger.info(str(len(feature_idxs_not_flipped)) + ' of ' + str(len(feature_names)) 
                + ' features have coefficients with the same signs in both years')
    return feature_idxs_not_flipped, feature_names_not_flipped

def run_nonstationarity_check_with_robust_features(X_csr_all_years_dict,
                                                   Y_all_years_dict,
                                                   feature_names,
                                                   orig_logreg_1,
                                                   orig_logreg_2,
                                                   orig_coef_df_1,
                                                   orig_coef_df_2,
                                                   output_file_header,
                                                   logger):
    '''
    Compare non-stationarity on logistic regressions with only features whose coefficients do not change as covariates
    vs non-stationarity on original logistic regressions
    @param X_csr_all_years_dict: dict mapping str to list of csr matrices, data split to covariate matrices in both years
    @param Y_all_years_dict: dict mapping str to list of np arrays, data split to outcomes in both years
    @param feature_names: list of str, feature names in covariate matrices
    @param orig_logreg_1: sklearn LogisticRegression, model for year 1
    @param orig_logreg_2: sklearn LogisticRegression, model for year 2
    @param orig_coef_df_1: pandas DataFrame, feature name and coefficient for year 1
    @param orig_coef_df_2: pandas DataFrame, feature name and coefficient for year 2
    @param output_file_header: str, start of path to saving new logistic regressions
    @param logger: logger, for INFO messages
    @return: None
    '''
    data_splits  = {'train', 'valid', 'test'}
    num_features = len(feature_names)
    assert data_splits.issubset(set(X_csr_all_years_dict.keys()))
    assert data_splits.issubset(set(Y_all_years_dict.keys()))
    assert np.all(np.array([len(X_csr_all_years_dict[data_split]) == 2
                            for data_split in data_splits]))
    assert np.all(np.array([len(Y_all_years_dict[data_split]) == 2
                            for data_split in data_splits]))
    for data_split in data_splits:
        assert np.all(np.array([X_csr_all_years_dict[data_split][year_idx].shape[0] == len(Y_all_years_dict[data_split][year_idx])
                                for year_idx in range(2)]))
        assert np.all(np.array([X_csr_all_years_dict[data_split][year_idx].shape[1] == num_features
                                for year_idx in range(2)]))
    assert orig_logreg_1.n_features_in_ == num_features
    assert orig_logreg_2.n_features_in_ == num_features
    assert len(orig_coef_df_1) == num_features
    assert len(orig_coef_df_2) == num_features
    
    # create X with only robust features
    start_time = time.time()
    robust_feature_idxs, robust_feature_names = select_robust_features(orig_coef_df_1,
                                                                       orig_coef_df_2,
                                                                       feature_names,
                                                                       logger)
    
    robust_X_csr_all_years_dict = {data_split: [csr_matrix(csc_matrix(X)[:, robust_feature_idxs])
                                                for X in X_csr_all_years_dict[data_split]]
                                   for data_split in X_csr_all_years_dict}
    logger.info('Time to create X with only robust features: ' + str(time.time() - start_time) + ' seconds')
    
    # learn logregs with robust features
    start_time = time.time()
    robust_logreg_1 = train_logreg(robust_X_csr_all_years_dict['train'][0],
                                   Y_all_years_dict['train'][0],
                                   robust_X_csr_all_years_dict['valid'][0],
                                   Y_all_years_dict['valid'][0],
                                   logger)
    save_logreg(robust_logreg_1,
                output_file_header + 'time2_logreg',
                robust_feature_names,
                logger)
    logger.info('Time to learn a logistic regression with only robust features in 2019: ' 
                + str(time.time() - start_time) + ' seconds')
    
    start_time = time.time()
    robust_logreg_2 = train_logreg(robust_X_csr_all_years_dict['train'][1],
                                   Y_all_years_dict['train'][1],
                                   robust_X_csr_all_years_dict['valid'][1],
                                   Y_all_years_dict['valid'][1],
                                   logger)
    save_logreg(robust_logreg_2,
                output_file_header + 'time3_logreg',
                robust_feature_names,
                logger)
    logger.info('Time to learn a logistic regression with only robust features in 2020: ' 
                + str(time.time() - start_time) + ' seconds')
    
    # evaluate logregs
    start_time = time.time()
    orig_logreg_1_test_pred_1   = orig_logreg_1.predict_proba(X_csr_all_years_dict['test'][0])[:,1]
    orig_logreg_1_test_auc_1    = roc_auc_score(Y_all_years_dict['test'][0],
                                                orig_logreg_1_test_pred_1)
    logger.info('Original year 1 logreg test AUC in year 1: ' + str(orig_logreg_1_test_auc_1))
    orig_logreg_1_test_pred_2   = orig_logreg_1.predict_proba(X_csr_all_years_dict['test'][1])[:,1]
    orig_logreg_1_test_auc_2    = roc_auc_score(Y_all_years_dict['test'][1],
                                                orig_logreg_1_test_pred_2)
    logger.info('Original year 1 logreg test AUC in year 2: ' + str(orig_logreg_1_test_auc_2))
    orig_logreg_2_test_pred_2   = orig_logreg_2.predict_proba(X_csr_all_years_dict['test'][1])[:,1]
    orig_logreg_2_test_auc_2    = roc_auc_score(Y_all_years_dict['test'][1],
                                                orig_logreg_2_test_pred_2)
    logger.info('Original year 2 logreg test AUC in year 2: ' + str(orig_logreg_2_test_auc_2))
    
    robust_logreg_1_test_pred_1 = robust_logreg_1.predict_proba(robust_X_csr_all_years_dict['test'][0])[:,1]
    robust_logreg_1_test_auc_1  = roc_auc_score(Y_all_years_dict['test'][0],
                                                robust_logreg_1_test_pred_1)
    logger.info('Robust features year 1 logreg test AUC in year 1: ' + str(robust_logreg_1_test_auc_1))
    robust_logreg_1_test_pred_2 = robust_logreg_1.predict_proba(robust_X_csr_all_years_dict['test'][1])[:,1]
    robust_logreg_1_test_auc_2  = roc_auc_score(Y_all_years_dict['test'][1],
                                                robust_logreg_1_test_pred_2)
    logger.info('Robust features year 1 logreg test AUC in year 2: ' + str(robust_logreg_1_test_auc_2))
    robust_logreg_2_test_pred_2 = robust_logreg_2.predict_proba(robust_X_csr_all_years_dict['test'][1])[:,1]
    robust_logreg_2_test_auc_2  = roc_auc_score(Y_all_years_dict['test'][1],
                                                robust_logreg_2_test_pred_2)
    logger.info('Robust features year 2 logreg test AUC in year 2: ' + str(robust_logreg_2_test_auc_2))
    logger.info('Time to evaluate logistic regressions: ' + str(time.time() - start_time) + ' seconds')
    
def run_nonstationarity_check_with_statsmodels_robust_features(X_csr_all_years_dict,
                                                               Y_all_years_dict,
                                                               feature_names,
                                                               outcome_name,
                                                               orig_logreg_1,
                                                               orig_logreg_2,
                                                               orig_coef_df_1,
                                                               orig_coef_df_2,
                                                               output_file_header,
                                                               year_idx,
                                                               logger):
    '''
    Compare non-stationarity on logistic regressions with only features whose coefficients do not change as covariates
    vs non-stationarity on original logistic regressions
    @param X_csr_all_years_dict: dict mapping str to list of csr matrices, data split to covariate matrices in both years
    @param Y_all_years_dict: dict mapping str to list of np arrays, data split to outcomes in both years
    @param feature_names: list of str, feature names in covariate matrices
    @param outcome_name: str, outcome name for statsmodels
    @param orig_logreg_1: statsmodels Logit, model for year 1
    @param orig_logreg_2: statsmodels Logit, model for year 2
    @param orig_coef_df_1: pandas DataFrame, feature name and 95% confidence interval for coefficient in year 1
    @param orig_coef_df_2: pandas DataFrame, feature name and 95% confidence interval for coefficient in year 2
    @param output_file_header: str, start of path to saving new logistic regressions
    @param logger: logger, for INFO messages
    @param year_idx: int, index of year 2
    @return: None
    '''
    selected_X_csr_all_years_dict, selected_feature_names = select_features_for_statsmodels(X_csr_all_years_dict,
                                                                                            Y_all_years_dict,
                                                                                            feature_names,
                                                                                            logger)
    
    orig_logreg_1_test_pred_1 = get_predictions_from_statsmodels_logreg(orig_logreg_1,
                                                                        selected_X_csr_all_years_dict['test'][0])
    orig_logreg_1_test_auc_1  = roc_auc_score(Y_all_years_dict['test'][0],
                                              orig_logreg_1_test_pred_1)
    logger.info('Original year 1 logreg test AUC in year 1: ' + str(orig_logreg_1_test_auc_1))
    
    orig_logreg_1_test_pred_2 = get_predictions_from_statsmodels_logreg(orig_logreg_1,
                                                                        selected_X_csr_all_years_dict['test'][1])
    orig_logreg_1_test_auc_2  = roc_auc_score(Y_all_years_dict['test'][1],
                                              orig_logreg_1_test_pred_2)
    logger.info('Original year 1 logreg test AUC in year 2: ' + str(orig_logreg_1_test_auc_2))
    
    orig_logreg_2_test_pred_2 = get_predictions_from_statsmodels_logreg(orig_logreg_2,
                                                                        selected_X_csr_all_years_dict['test'][1])
    orig_logreg_2_test_auc_2  = roc_auc_score(Y_all_years_dict['test'][1],
                                              orig_logreg_2_test_pred_2)
    logger.info('Original year 2 logreg test AUC in year 2: ' + str(orig_logreg_2_test_auc_2))
    
    robust_feature_idxs = np.argwhere(np.logical_or(np.logical_and(orig_coef_df_1['[0.025'].values[:-1] > 0,
                                                                   orig_coef_df_2['[0.025'].values[:-1] > 0),
                                                    np.logical_and(orig_coef_df_1['0.975]'].values[:-1] < 0,
                                                                   orig_coef_df_2['0.975]'].values[:-1] < 0))).flatten()
    logger.info(str(len(robust_feature_idxs)) + ' of ' + str(len(selected_feature_names)) + ' features are robust')
    robust_X_csr_all_years_dict = {data_split: [csr_matrix(csc_matrix(X)[:,robust_feature_idxs])
                                                for X in selected_X_csr_all_years_dict[data_split]]
                                   for data_split in selected_X_csr_all_years_dict}
    robust_feature_names        = [feature_names[idx] for idx in robust_feature_idxs]
    
    robust_logreg_year1_filename = output_file_header + str(year_idx - 1) + '.pkl'
    if os.path.exists(robust_logreg_year1_filename):
        logger.info('Loading model from ' + robust_logreg_year1_filename)
        robust_logreg_year1 = sm.load(robust_logreg_year1_filename)
    else:
        robust_logreg_year1 = train_statsmodels_logreg(robust_X_csr_all_years_dict['train'][0],
                                                       Y_all_years_dict['train'][0],
                                                       robust_feature_names,
                                                       outcome_name,
                                                       logger)
        save_statsmodels_logreg(robust_logreg_year1,
                                output_file_header + str(year_idx - 1),
                                logger)
    
    robust_logreg_year2_filename = output_file_header + str(year_idx) + '.pkl'
    if os.path.exists(robust_logreg_year2_filename):
        logger.info('Loading model from ' + robust_logreg_year2_filename)
        robust_logreg_year2 = sm.load(robust_logreg_year2_filename)
    else:
        robust_logreg_year2 = train_statsmodels_logreg(robust_X_csr_all_years_dict['train'][1],
                                                       Y_all_years_dict['train'][1],
                                                       robust_feature_names,
                                                       outcome_name,
                                                       logger)
        save_statsmodels_logreg(robust_logreg_year2,
                                output_file_header + str(year_idx),
                                logger)
    
    robust_logreg_1_test_pred_1 = get_predictions_from_statsmodels_logreg(robust_logreg_year1,
                                                                          robust_X_csr_all_years_dict['test'][0])
    robust_logreg_1_test_auc_1  = roc_auc_score(Y_all_years_dict['test'][0],
                                                robust_logreg_1_test_pred_1)
    logger.info('Robust year 1 logreg test AUC in year 1: ' + str(robust_logreg_1_test_auc_1))
    
    robust_logreg_1_test_pred_2 = get_predictions_from_statsmodels_logreg(robust_logreg_year1,
                                                                          robust_X_csr_all_years_dict['test'][1])
    robust_logreg_1_test_auc_2  = roc_auc_score(Y_all_years_dict['test'][1],
                                                robust_logreg_1_test_pred_2)
    logger.info('Robust year 1 logreg test AUC in year 2: ' + str(robust_logreg_1_test_auc_2))
    
    robust_logreg_2_test_pred_2 = get_predictions_from_statsmodels_logreg(robust_logreg_year2,
                                                                          robust_X_csr_all_years_dict['test'][1])
    robust_logreg_2_test_auc_2  = roc_auc_score(Y_all_years_dict['test'][1],
                                                robust_logreg_2_test_pred_2)
    logger.info('Robust year 2 logreg test AUC in year 2: ' + str(robust_logreg_2_test_auc_2))
    
def create_parser():
    '''
    Create an argument parser
    @return: argparse ArgumentParser
    '''
    parser = argparse.ArgumentParser(description = ('Examine addressing conditional shift '
                                                    'by selecting robust features'))
    parser.add_argument('--outcome_name',
                        action = 'store',
                        type   = str,
                        help   = 'Specify outcome to examine.')
    parser.add_argument('--statsmodels',
                        action  = 'store_true',
                        default = False,
                        help    = 'Specify whether to use statsmodels Logit or sklearn LogisticRegression.')
    return parser
    
if __name__ == '__main__':
    
    mp.set_start_method('spawn', force=True)
    
    parser = create_parser()
    args   = parser.parse_args()
    
    outcome_name_to_type_id = {'Inpatient consultation': 'procedure_inpatient_consultation',
                               'Nursing care'          : 'procedure_nursing'}
    
    assert args.outcome_name in outcome_name_to_type_id
    
    outcome_name        = outcome_name_to_type_id[args.outcome_name] + '_outcomes'
    if outcome_name.startswith('condition_'):
        freq_str        = 'freq100'
    else:
        freq_str        = 'freq300'
    
    outcome_name_to_year_idx = {'Inpatient consultation': 4,
                                'Nursing care'          : 1}
    curr_year = outcome_name_to_year_idx[args.outcome_name]
    prev_year = curr_year - 1
    
    experiment_name        = outcome_name + '_from_all_' + freq_str + '_logreg'
    experiment_dir         = config.experiment_dir + experiment_name + '/'
    dataset_file_header    = config.outcome_data_dir + 'dataset_' + outcome_name + '/fold0_' + freq_str
    if args.statsmodels:
        logreg_file_header = experiment_dir + experiment_name + '_statsmodels_year' + str(curr_year) + 'v' + str(prev_year) \
                           + '_time'
        output_file_header = experiment_dir + experiment_name + '_statsmodels_year' + str(curr_year) + 'v' + str(prev_year) \
                           + '_robust_features_'
    else:
        logreg_file_header = experiment_dir + experiment_name + '_logistic_regression_time'
        output_file_header = experiment_dir + experiment_name + '_robust_features_'
    
    logging_filename = experiment_dir + 'select_robust_features_for_conditional_shift_' \
                     + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.log'
    logger           = set_up_logger('logger_main',
                                     logging_filename)
    multiprocessing_logging.install_mp_handler()
    logger.info('python3 select_robust_features_for_conditional_shift.py'
                + ' --outcome_name=' + args.outcome_name
                + ' --statsmodels='  + str(args.statsmodels))
    
    
    # load data
    this_start_time  = time.time()
    Y_all_years_dict = load_outcomes(dataset_file_header,
                                     num_years         = 2,
                                     logger            = logger,
                                     starting_year_idx = prev_year)
    logger.info('Time to load outcomes: ' + str(time.time() - this_start_time) + ' seconds')
    
    this_start_time                        = time.time()
    X_csr_all_years_dict, feature_names, _ = load_covariates(dataset_file_header,
                                                             'all',
                                                             num_years         = 2,
                                                             logger            = logger,
                                                             starting_year_idx = prev_year)
    logger.info('Time to load covariates: ' + str(time.time() - this_start_time) + ' seconds')
    
    # load logistic regressions and coefficients
    if args.statsmodels:
        orig_logreg_1 = sm.load(logreg_file_header + str(prev_year) + '.pkl')
        orig_logreg_2 = sm.load(logreg_file_header + str(curr_year) + '.pkl')
    else:
        orig_logreg_1 = joblib.load(logreg_file_header + str(prev_year) + '.joblib')
        orig_logreg_2 = joblib.load(logreg_file_header + str(curr_year) + '.joblib')
    orig_coef_df_1    = pd.read_csv(logreg_file_header + str(prev_year) + '_coefficients.csv')
    orig_coef_df_2    = pd.read_csv(logreg_file_header + str(curr_year) + '_coefficients.csv')
    
    if args.statsmodels:
        run_nonstationarity_check_with_statsmodels_robust_features(X_csr_all_years_dict,
                                                                   Y_all_years_dict,
                                                                   feature_names,
                                                                   args.outcome_name,
                                                                   orig_logreg_1,
                                                                   orig_logreg_2,
                                                                   orig_coef_df_1,
                                                                   orig_coef_df_2,
                                                                   output_file_header,
                                                                   curr_year,
                                                                   logger)
    else:
        run_nonstationarity_check_with_robust_features(X_csr_all_years_dict,
                                                       Y_all_years_dict,
                                                       feature_names,
                                                       orig_logreg_1,
                                                       orig_logreg_2,
                                                       orig_coef_df_1,
                                                       orig_coef_df_2,
                                                       output_file_header,
                                                       logger)