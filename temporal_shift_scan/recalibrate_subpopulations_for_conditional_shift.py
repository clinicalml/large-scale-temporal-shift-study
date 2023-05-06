import os
import sys
import argparse
import joblib
import numpy as np
from os.path import dirname, abspath, join
from sklearn.metrics import roc_auc_score
from datetime import datetime
from scipy.sparse import csc_matrix

from load_data_for_nonstationarity_scan import load_outcomes, load_covariates

sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from logging_utils import set_up_logger

def create_subpopulation_function(feature_names,
                                  subpop_feature_names,
                                  subpop_feature_relation,
                                  logger):
    '''
    Create a function that returns indices of samples in sub-population defined by features
    @param feature_names: list of str, names of feature in each column of covariate matrix
    @param subpop_feature_names: list of str, names of features for defining sub-populations
    @param subpop_feature_relation: str, 'AND' or 'OR', whether all features need to be present for sub-population or only 1
    @param logger: logger, for INFO messages
    @return: function that takes in X and returns indices
    '''
    assert subpop_feature_relation in {'AND', 'OR'}
    subpop_feature_idxs = [feature_names.index(feat_name)
                           for feat_name in subpop_feature_names]
    logger.info('Sub-population is defined as ' + (' ' + subpop_feature_relation + ' ').join(subpop_feature_names))
    def get_subpopulation_idxs(X):
        '''
        Get indices of samples that are in a specific sub-population
        @param X: csr matrix, # samples x # features, sample features
        @return: np array, indices of samples
        '''
        if len(subpop_feature_idxs) == 1:
            return np.nonzero(csc_matrix(X)[:,subpop_feature_idxs[0]].toarray().flatten())[0]
        if subpop_feature_relation == 'OR':
            return np.nonzero(np.asarray(csc_matrix(X)[:,subpop_feature_idxs].sum(axis = 1)).flatten())[0]
        if subpop_feature_relation == 'AND':
            return np.nonzero(np.asarray(csc_matrix(X)[:,subpop_feature_idxs].sum(axis = 1)).flatten() 
                              == len(subpop_feature_idxs))[0]
    return get_subpopulation_idxs

def compute_subpopulation_outcome_frequency(X,
                                            Y,
                                            get_subpopulation_idxs,
                                            logger):
    '''
    Compute outcome frequency within sub-population in training and validation splits
    @param X: dict mapping str to csr matrix, # samples x # features, data split to sample features
    @param Y: dict mapping str to np array, # samples, data split to sample outcomes
    @param get_subpopulation_idxs: function that takes in X and returns indices
    @param logger: logger, for INFO messages
    @return: float
    '''
    data_splits = {'train', 'valid'}
    assert data_splits.issubset(set(X.keys()))
    assert data_splits.issubset(set(Y.keys()))
    assert np.all(np.array([X[data_split].shape[0] == len(Y[data_split])
                            for data_split in data_splits]))
    
    train_subpop_idxs   = get_subpopulation_idxs(X['train'])
    train_outcome_count = Y['train'][train_subpop_idxs].sum()
    valid_subpop_idxs   = get_subpopulation_idxs(X['valid'])
    valid_outcome_count = Y['valid'][valid_subpop_idxs].sum()
    total_outcome_count = train_outcome_count + valid_outcome_count
    total_subpop_count  = len(train_subpop_idxs) + len(valid_subpop_idxs)
    outcome_prob        = float(total_outcome_count)/total_subpop_count
    logger.info('Sub-population outcome frequency: ' + str(outcome_prob) + ', ' 
                + str(total_outcome_count) + ' of ' + str(total_subpop_count))
    return outcome_prob

def reverse_subpopulation_function(get_subpopulation_idxs):
    '''
    Create a function that returns indices that are not selected by get_subpopulation_idxs
    @param get_subpopulation_idxs: function that takes in X and returns indices in sub-population
    @return: function that takes in X and returns indices not in sub-population
    '''
    def get_reverse_subpopulation_idxs(X):
        '''
        Get indices of samples that are not in a specific sub-population
        @param X: csr matrix, # samples x # features, sample features
        @return: np array, indices of samples
        '''
        subpop_idxs = get_subpopulation_idxs(X)
        bool_mask   = np.ones(X.shape[0], 
                              dtype = bool)
        bool_mask[subpop_idxs] = False
        return np.nonzero(bool_mask)[0]
    return get_reverse_subpopulation_idxs

def get_multiplicative_recalibration_weight(get_subpopulation_idxs,
                                            X_source,
                                            Y_source,
                                            X_target,
                                            Y_target,
                                            logger):
    '''
    Compute ratio between outcome frequencies in target vs source as recalibration weight within that sub-population
    @param get_subpopulation_idxs: function that takes in X and returns indices in sub-population
    @param X_source: dict mapping str to csr matrix, # samples x # features, data split to sample features in source domain
    @param Y_source: dict mapping str to np array, # samples, data split to sample outcomes in source domain
    @param X_target: dict mapping str to csr matrix, # samples x # features, data split to sample features in target domain
    @param Y_target: dict mapping str to np array, # samples, data split to sample outcomes in target domain
    @param logger: logger, for INFO messages
    @return: float, recalibration weight
    '''
    data_splits = {'train', 'valid'}
    assert data_splits.issubset(set(X_source.keys()))
    assert data_splits.issubset(set(Y_source.keys()))
    assert data_splits.issubset(set(X_target.keys()))
    assert data_splits.issubset(set(Y_target.keys()))
    num_features = X_source['train'].shape[1]
    assert np.all(np.array([X_source[data_split].shape[1] == num_features
                            for data_split in data_splits]))
    assert np.all(np.array([X_target[data_split].shape[1] == num_features
                            for data_split in data_splits]))
    assert np.all(np.array([X_source[data_split].shape[0] == len(Y_source[data_split])
                            for data_split in data_splits]))
    assert np.all(np.array([X_target[data_split].shape[0] == len(Y_target[data_split])
                            for data_split in data_splits]))
    
    logger.info('Computing sub-population outcome frequency in source domain')
    source_outcome_prob = compute_subpopulation_outcome_frequency(X_source,
                                                                  Y_source,
                                                                  get_subpopulation_idxs,
                                                                  logger)
    
    logger.info('Computing sub-population outcome frequency in target domain')
    target_outcome_prob = compute_subpopulation_outcome_frequency(X_target,
                                                                  Y_target,
                                                                  get_subpopulation_idxs,
                                                                  logger)
    
    if source_outcome_prob == 0:
        if target_outcome_prob == 0:
            recalibration_weight = 1
        else:
            recalibration_weight = 2
    else:
        recalibration_weight = target_outcome_prob/source_outcome_prob
    logger.info('Multiplicative re-calibration weight: ' + str(recalibration_weight))
    
    return recalibration_weight

def check_for_nonstationarity_with_recalibration(source_model,
                                                 target_model,
                                                 outcome_name,
                                                 feature_names,
                                                 subpop_feature_names,
                                                 subpop_feature_relations,
                                                 X_source,
                                                 Y_source,
                                                 X_target,
                                                 Y_target,
                                                 logger):
    '''
    Re-calibrate predictions from source model for target samples within sub-population 
    Multiplicative re-calibration weight is ratio between sub-population label frequencies in target vs source domains
    @param source_model: model with predict_proba function, for source domain
    @param target_model: model with predict_proba function, for target domain
    @param outcome_name: str, name of outcome to examine
    @param feature_names: list of str, names of feature in each column of covariate matrix
    @param subpop_feature_names: list of lists of str, names of features for defining sub-populations,
                                 each inner list defines a single sub-population
    @param subpop_feature_relations: list of str, 'AND' or 'OR', whether all features or only 1 feature needs to be present 
                                     for a sample to be in each sub-population
    @param X_source: dict mapping str to csr matrix, # samples x # features, data split to sample features in source domain
    @param Y_source: dict mapping str to np array, # samples, data split to sample outcomes in source domain
    @param X_target: dict mapping str to csr matrix, # samples x # features, data split to sample features in target domain
    @param Y_target: dict mapping str to np array, # samples, data split to sample outcomes in target domain
    @param logger: logger, for INFO messages
    @return: np array, # samples, predicted probabilities for each sample
    '''
    num_features = len(feature_names)
    assert source_model.n_features_in_ == num_features
    assert target_model.n_features_in_ == num_features
    source_data_splits = {'train', 'valid'}
    target_data_splits = {'train', 'valid', 'test'}
    assert source_data_splits.issubset(set(X_source.keys()))
    assert source_data_splits.issubset(set(Y_source.keys()))
    assert target_data_splits.issubset(set(X_target.keys()))
    assert target_data_splits.issubset(set(Y_target.keys()))
    assert np.all(np.array([X_source[data_split].shape[1] == num_features
                            for data_split in source_data_splits]))
    assert np.all(np.array([X_target[data_split].shape[1] == num_features
                            for data_split in target_data_splits]))
    assert np.all(np.array([X_source[data_split].shape[0] == len(Y_source[data_split])
                            for data_split in source_data_splits]))
    assert np.all(np.array([X_target[data_split].shape[0] == len(Y_target[data_split])
                            for data_split in target_data_splits]))
    assert len(subpop_feature_names) > 0
    assert len(subpop_feature_names) == len(subpop_feature_relations)
    for subpop_idx in range(len(subpop_feature_names)):
        assert len(subpop_feature_names[subpop_idx]) > 0
        assert subpop_feature_relations[subpop_idx] in {'AND', 'OR'}
    
    # get re-calibration weight within sub-population
    recalibration_subpop_functions = dict()
    recalibration_subpop_weights   = dict()
    multiplicative_groups          = []
    for subpop_idx in range(len(subpop_feature_names)):
        get_subpopulation_idxs = create_subpopulation_function(feature_names,
                                                               subpop_feature_names[subpop_idx],
                                                               subpop_feature_relations[subpop_idx],
                                                               logger)
        subpop_name            = 'subpop' + str(subpop_idx)

        recalibration_weight = get_multiplicative_recalibration_weight(get_subpopulation_idxs,
                                                                       X_source,
                                                                       Y_source,
                                                                       X_target,
                                                                       Y_target,
                                                                       logger)
        multiplicative_groups.append(subpop_name)

        recalibration_subpop_functions[subpop_name] = get_subpopulation_idxs
        recalibration_subpop_weights[subpop_name]   = recalibration_weight
    
    # re-calibrate target predictions
    target_test_pred = source_model.predict_proba(X_target['test'])[:,1]
    target_test_auc  = roc_auc_score(Y_target['test'],
                                     target_test_pred)
    logger.info('AUC of source model on target data: ' + str(target_test_auc))
    
    target_test_pred_recalibrated = target_test_pred.copy()
    for group in multiplicative_groups:
        target_test_subpop_idxs   = recalibration_subpop_functions[group](X_target['test'])
        target_test_pred_recalibrated[target_test_subpop_idxs] *= recalibration_subpop_weights[group]
    target_test_pred_recalibrated = np.clip(target_test_pred_recalibrated, 0, 1)
    target_test_auc_recalibrated  = roc_auc_score(Y_target['test'],
                                                  target_test_pred_recalibrated)
    logger.info('AUC of source model on target data re-calibrated within sub-population: ' + str(target_test_auc_recalibrated))
    
    target_oracle_test_pred = target_model.predict_proba(X_target['test'])[:,1]
    target_oracle_test_auc  = roc_auc_score(Y_target['test'],
                                            target_oracle_test_pred)
    logger.info('AUC of target model on target data: ' + str(target_oracle_test_auc))

def create_parser():
    '''
    Create an argument parser
    @return: argparse ArgumentParser
    '''
    parser = argparse.ArgumentParser(description = ('Examine addressing conditional shift '
                                                    'by reweighting predicted probabilities'))
    parser.add_argument('--outcome_name',
                        action = 'store',
                        type   = str,
                        help   = 'Specify outcome to examine.')
    return parser

if __name__ == '__main__':
    
    parser = create_parser()
    args   = parser.parse_args()
    
    outcome_name_to_type_id = {'Inpatient consultation'           : 'procedure_inpatient_consultation',
                               'Nursing care'                     : 'procedure_nursing'}
    
    assert args.outcome_name in outcome_name_to_type_id
    
    if outcome_name_to_type_id[args.outcome_name].startswith('condition'):
        freq_str        = 'freq100'
    else:
        freq_str        = 'freq300'
    experiment_name     = outcome_name_to_type_id[args.outcome_name] + '_outcomes_from_all_' + freq_str + '_logreg'
    dataset_name        = 'dataset_' + outcome_name_to_type_id[args.outcome_name] + '_outcomes'
    experiment_dir      = config.experiment_dir + experiment_name + '/'
    dataset_file_header = config.outcome_data_dir + dataset_name + '/fold0_' + freq_str
    
    logging_filename    = experiment_dir + experiment_name + '_recalibrate_subpopulations_for_conditional_shift_' \
                        + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.log'
    logger              = set_up_logger('logger_main',
                                        logging_filename)
    logger.info('python3 recalibrate_subpopulations_for_conditional_shift.py --outcome_name=\"' + args.outcome_name + '\"')
    
    outcome_name_to_year_idx = {'Inpatient consultation'           : 4,
                                'Nursing care'                     : 1}
    
    source_model = joblib.load(experiment_dir + experiment_name + '_logistic_regression_time' 
                               + str(outcome_name_to_year_idx[args.outcome_name] - 1) + '.joblib')
    target_model = joblib.load(experiment_dir + experiment_name + '_logistic_regression_time'
                               + str(outcome_name_to_year_idx[args.outcome_name]) + '.joblib')
    
    Xs, feature_names, _ = load_covariates(dataset_file_header,
                                           feature_set       = 'all',
                                           num_years         = 2,
                                           logger            = logger,
                                           starting_year_idx = outcome_name_to_year_idx[args.outcome_name] - 1)
    X_source             = {data_split: Xs[data_split][0]
                            for data_split in Xs}
    X_target             = {data_split: Xs[data_split][1]
                            for data_split in Xs}
    Ys                   = load_outcomes(dataset_file_header,
                                         num_years           = 2,
                                         logger              = logger,
                                         starting_year_idx   = outcome_name_to_year_idx[args.outcome_name] - 1)
    Y_source             = {data_split: Ys[data_split][0]
                            for data_split in Ys}
    Y_target             = {data_split: Ys[data_split][1]
                            for data_split in Ys}
    
    outcome_to_subpop_feature_names = {'Inpatient consultation': 
                                       [['319835 - condition - Congestive heart failure - 30 days'],
                                        [('764123 - condition - Atherosclerosis of coronary artery without angina pectoris '
                                          '- 30 days')]],
                                       'Nursing care': 
                                       [[('2514464 - procedure - Nursing facility discharge day management; '
                                          '30 minutes or less - 30 days'), 
                                         ('2514465 - procedure - Nursing facility discharge day management; '
                                          'more than 30 minutes - 30 days')]]}
    outcome_to_subpop_relations     = {'Inpatient consultation': ['AND', 'AND'],
                                       'Nursing care': ['OR']}
    
    check_for_nonstationarity_with_recalibration(source_model,
                                                 target_model,
                                                 args.outcome_name,
                                                 feature_names,
                                                 outcome_to_subpop_feature_names[args.outcome_name],
                                                 outcome_to_subpop_relations[args.outcome_name],
                                                 X_source,
                                                 Y_source,
                                                 X_target,
                                                 Y_target,
                                                 logger)