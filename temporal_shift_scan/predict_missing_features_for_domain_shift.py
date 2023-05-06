import os
import sys
import argparse
import joblib
import numpy as np
from copy import deepcopy
from os.path import dirname, abspath, join
from scipy.sparse import csr_matrix, csc_matrix, hstack, vstack
from sklearn.metrics import roc_auc_score
from datetime import datetime

from load_data_for_nonstationarity_scan import load_outcomes, load_covariates

sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from logging_utils import set_up_logger
from model_utils import train_logreg, save_logreg, eval_predictions

def predict_missing_features(X_source_30days,
                             X_target_30days,
                             feature_names_30days,
                             important_feature_names,
                             X_source_365days,
                             X_target_365days,
                             feature_names_365days,
                             output_dir,
                             logger):
    '''
    Learn logistic regressions to predict important 30-day features from 365-day features using source data
    For target samples without feature, impute missing features with predictions
    @param X_source_30days: dict mapping str to csr matrix, data split to feature matrix containing source data, 
                            features are indicators for whether they occurred in past 30 days
    @param X_target_30days: dict mapping str to csr matrix, data split to feature matrix containing target data,
                            features are indicators for whether they occurred in past 30 days
    @param feature_names_30days: list of str, names of features in each column of feature matrices above
    @param important_feature_names: list of str, names of important features to impute
    @param X_source_365days: dict mapping str to csr matrix, data split to feature matrix containing source data, 
                             features are indicators for whether they occurred in past 365 days
    @param X_target_365days: dict mapping str to csr matrix, data split to feature matrix containing target data,
                             features are indicators for whether they occurred in past 365 days
    @param feature_names_365days: list of str, names of features in each column of feature matrices above
    @param output_dir: str, path to directory in which to save logistic regressions
    @param logger: logger, for INFO messages
    @return: 1. dict mapping str to csr matrix, data split to imputed 30-day feature matrix for source data
             2. dict mapping str to csr matrix, data split to imputed 30-day feature matrix for target data
    '''
    assert {'train', 'valid'}.issubset(set(X_source_30days.keys()))
    assert {'train', 'valid'}.issubset(set(X_source_365days.keys()))
    assert X_target_30days.keys() == X_target_365days.keys()
    num_features_30days = len(feature_names_30days)
    assert np.all(np.array([X_source_30days[data_split].shape[1] == num_features_30days
                            for data_split in X_source_30days.keys()]))
    assert np.all(np.array([X_target_30days[data_split].shape[1] == num_features_30days
                            for data_split in X_target_30days.keys()]))
    num_features_365days = len(feature_names_365days)
    assert np.all(np.array([X_source_365days[data_split].shape[1] == num_features_365days
                            for data_split in X_source_365days.keys()]))
    assert np.all(np.array([X_target_365days[data_split].shape[1] == num_features_365days
                            for data_split in X_target_365days.keys()]))
    assert np.all(np.array([X_source_30days[data_split].shape[0] == X_source_365days[data_split].shape[0]
                            for data_split in X_source_365days.keys()]))
    assert np.all(np.array([X_target_30days[data_split].shape[0] == X_target_365days[data_split].shape[0]
                            for data_split in X_target_365days.keys()]))
    
    X_source_30days_csc      = {data_split: csc_matrix(X_source_30days[data_split])
                                for data_split in X_source_30days}
    X_target_30days_csc      = {data_split: csc_matrix(X_target_30days[data_split])
                                for data_split in X_target_30days}
    important_feature_idxs   = [feature_names_30days.index(important_feature_names[idx])
                                for idx in range(len(important_feature_names))]
    
    imputed_source_data      = deepcopy(X_source_30days_csc)
    imputed_target_data      = deepcopy(X_target_30days_csc)
    for idx in range(len(important_feature_names)):
        # Compute feature frequencies
        important_feature_name = important_feature_names[idx]
        logger.info('Examining feature ' + important_feature_name)
        feature_idx          = important_feature_idxs[idx]
        num_source_samples   = sum([X_source_30days_csc[data_split].shape[0]
                                    for data_split in X_source_30days_csc])
        source_feature_count = sum([X_source_30days_csc[data_split][:,feature_idx].sum()
                                    for data_split in X_source_30days_csc])
        logger.info('Source frequency: ' + str(float(source_feature_count)/num_source_samples))
        
        num_target_samples   = sum([X_target_30days_csc[data_split].shape[0]
                                    for data_split in X_target_30days_csc])
        target_feature_count = sum([X_target_30days_csc[data_split][:,feature_idx].sum()
                                    for data_split in X_target_30days_csc])
        logger.info('Target frequency: ' + str(float(target_feature_count)/num_target_samples))
        
        # Learn a logistic regression predicting feature
        feature_name_parts   = important_feature_name.split(' - ')
        feature_type         = feature_name_parts[1]
        feature_id           = feature_name_parts[0]
        missing_feature_logreg_file_header = output_dir + 'impute_' + feature_type + '_' + feature_id \
                                           + '_feature_time2_logreg'
        missing_feature_logreg_filename    = missing_feature_logreg_file_header + '.joblib'
        if os.path.exists(missing_feature_logreg_filename):
            feature_logreg                 = joblib.load(missing_feature_logreg_filename)
        else:
            feature_source_train           = X_source_30days_csc['train'][:,feature_idx].toarray().flatten()
            feature_source_valid           = X_source_30days_csc['valid'][:,feature_idx].toarray().flatten()
            
            feature_logreg                 = train_logreg(X_source_365days['train'],
                                                          feature_source_train,
                                                          X_source_365days['valid'],
                                                          feature_source_valid,
                                                          logger)
            save_logreg(feature_logreg,
                        missing_feature_logreg_file_header,
                        feature_names_365days,
                        logger)
       
        # Impute source data
        for data_split in X_source_30days_csc:
            feature_source = X_source_30days_csc[data_split][:,feature_idx].toarray().flatten()
            imputed_source_feature = csc_matrix(np.where(feature_source == 1, 
                                                         1, 
                                                         feature_logreg.predict(X_source_365days[data_split])).reshape((-1,1)))
            imputed_source_data[data_split] = hstack((imputed_source_data[data_split][:,:feature_idx],
                                                      imputed_source_feature,
                                                      imputed_source_data[data_split][:,feature_idx+1:]),
                                                     format = 'csr')
        imputed_source_feature_count = sum([imputed_source_data[data_split][:,feature_idx].sum()
                                            for data_split in imputed_source_data])
        logger.info('Imputed source frequency: ' + str(float(imputed_source_feature_count)/num_source_samples))
    
        # Impute target data
        for data_split in X_target_30days_csc:
            feature_target = X_target_30days_csc[data_split][:,feature_idx].toarray().flatten()
            
            imputed_target_feature = csc_matrix(np.where(feature_target == 1, 
                                                         1, 
                                                         feature_logreg.predict(X_target_365days[data_split])).reshape((-1,1)))
            imputed_target_data[data_split] = hstack((imputed_target_data[data_split][:,:feature_idx],
                                                      imputed_target_feature,
                                                      imputed_target_data[data_split][:,feature_idx+1:]),
                                                     format = 'csr')
        imputed_target_feature_count = sum([imputed_target_data[data_split][:,feature_idx].sum()
                                            for data_split in imputed_target_data])
        logger.info('Imputed target frequency: ' + str(float(imputed_target_feature_count)/num_target_samples))
    return imputed_source_data, imputed_target_data

def check_for_nonstationarity_with_imputed_features(source_model,
                                                    target_model,
                                                    X_source_30days,
                                                    X_target_30days,
                                                    Y_source,
                                                    Y_target,
                                                    feature_names_30days,
                                                    important_feature_names,
                                                    X_source_365days,
                                                    X_target_365days,
                                                    feature_names_365days,
                                                    output_dir,
                                                    logger):
    '''
    Check for non-stationarity when evaluating previous year's model on current year's data with imputed features
    @param source_model: model with predict_proba function, model for previous year
    @param target_model: model with predict_proba function, model for current year
    @param X_source_30days: dict mapping str to csr matrix, # samples x # features, 
                            data split to sample features for previous year, 
                            features are indicators for whether they occurred in past 30 days
    @param X_target_30days: dict mapping str to csr matrix, # samples x # features, 
                            data split to sample features for current year, 
                            features are indicators for whether they occurred in past 30 days
    @param Y_source: dict mapping str to np array, # samples, data split to sample outcomes for previous year
    @param Y_target: dict mapping str to np array, # samples, data split to sample outcomes for current year
    @param feature_names_30days: list of str, names of features in each column of feature matrices above
    @param important_feature_names: list of str, names of features to impute
    @param X_source_365days: dict mapping str to csr matrix, # samples x # features, 
                             data split to sample features for previous year, 
                             features are indicators for whether they occurred in past 365 days
    @param X_target_365days: dict mapping str to csr matrix, # samples x # features, 
                             data split to sample features for current year, 
                             features are indicators for whether they occurred in past 365 days
    @param feature_names_365days: list of str, names of features in each column of feature matrices above
    @param output_dir: str, path to directory in which to save logistic regressions
    @param logger: logger, for INFO messages
    @return: None
    '''
    source_data_splits = {'train', 'valid', 'test'}
    target_data_splits = {'test'}
    assert source_data_splits.issubset(X_source_30days.keys())
    assert target_data_splits.issubset(X_target_30days.keys())
    assert source_data_splits.issubset(X_source_365days.keys())
    assert target_data_splits.issubset(X_target_365days.keys())
    assert target_data_splits.issubset(Y_target.keys())
    num_features_30days = len(feature_names_30days)
    assert source_model.n_features_in_ == num_features_30days
    assert target_model.n_features_in_ == num_features_30days
    assert np.all(np.array([X_source_30days[data_split].shape[1] == num_features_30days
                            for data_split in source_data_splits]))
    assert np.all(np.array([X_target_30days[data_split].shape[1] == num_features_30days
                            for data_split in target_data_splits]))
    assert np.all(np.array([X_source_30days[data_split].shape[0] == len(Y_source[data_split])
                            for data_split in source_data_splits]))
    assert np.all(np.array([X_target_30days[data_split].shape[0] == len(Y_target[data_split])
                            for data_split in target_data_splits]))
    num_features_365days = len(feature_names_365days)
    assert np.all(np.array([X_source_365days[data_split].shape[1] == num_features_365days
                            for data_split in source_data_splits]))
    assert np.all(np.array([X_target_365days[data_split].shape[1] == num_features_365days
                            for data_split in target_data_splits]))
    assert np.all(np.array([X_source_365days[data_split].shape[0] == len(Y_source[data_split])
                            for data_split in source_data_splits]))
    assert np.all(np.array([X_target_365days[data_split].shape[0] == len(Y_target[data_split])
                            for data_split in target_data_splits]))
    
    X_source_imputed, X_target_imputed = predict_missing_features(X_source_30days,
                                                                  X_target_30days,
                                                                  feature_names_30days,
                                                                  important_feature_names,
                                                                  X_source_365days,
                                                                  X_target_365days,
                                                                  feature_names_365days,
                                                                  output_dir,
                                                                  logger)
    
    logger.info('Evaluating 2019 model on 2019 test data')
    source_pred         = source_model.predict_proba(X_source_30days['test'])[:,1]
    eval_predictions(Y_source['test'],
                     source_pred,
                     logger)
    
    logger.info('Evaluating 2019 model on 2019 imputed test data')
    source_pred_imputed = source_model.predict_proba(X_source_imputed['test'])[:,1]
    eval_predictions(Y_source['test'],
                     source_pred_imputed,
                     logger)
    
    logger.info('Evaluating 2019 model on 2020 test data')
    target_pred         = source_model.predict_proba(X_target_30days['test'])[:,1]
    eval_predictions(Y_target['test'],
                     target_pred,
                     logger)
    
    logger.info('Evaluating 2019 model on 2020 imputed test data')
    target_pred_imputed = source_model.predict_proba(X_target_imputed['test'])[:,1]
    eval_predictions(Y_target['test'],
                     target_pred_imputed,
                     logger)
        
def evaluate_oracle_imputed_features(target_model,
                                     X_target_30days,
                                     Y_target,
                                     feature_names_30days,
                                     important_feature_names,
                                     X_target_365days,
                                     feature_names_365days,
                                     output_dir,
                                     logger):
    '''
    Learn logistic regressions to predict important 30-day features from 365-day features using target data
    Evaluate predictions from target model using imputed features
    @param target_model: model with predict_proba function, model for current year
    @param X_target_30days: dict mapping str to csr matrix, # samples x # features, 
                            data split to sample features for current year, 
                            features are indicators for whether they occurred in past 30 days
    @param Y_target: dict mapping str to np array, # samples, data split to sample outcomes for current year
    @param feature_names_30days: list of str, names of features in each column of feature matrices above
    @param important_feature_names: list of str, names of features to impute
    @param X_target_365days: dict mapping str to csr matrix, # samples x # features, 
                             data split to sample features for current year, 
                             features are indicators for whether they occurred in past 365 days
    @param feature_names_365days: list of str, names of features in each column of feature matrices above
    @param output_dir: str, path to directory in which to save logistic regressions
    @param logger: logger, for INFO messages
    @return: None
    '''
    logger.info('Oracle version: Feature imputation models are learned on 2020 data.')
    target_data_splits = {'train', 'valid', 'test'}
    assert target_data_splits.issubset(X_target_30days.keys())
    assert target_data_splits.issubset(X_target_365days.keys())
    assert target_data_splits.issubset(Y_target.keys())
    num_features_30days = len(feature_names_30days)
    assert target_model.n_features_in_ == num_features_30days
    assert np.all(np.array([X_target_30days[data_split].shape[1] == num_features_30days
                            for data_split in target_data_splits]))
    assert np.all(np.array([X_target_30days[data_split].shape[0] == len(Y_target[data_split])
                            for data_split in target_data_splits]))
    num_features_365days = len(feature_names_365days)
    assert np.all(np.array([X_target_365days[data_split].shape[1] == num_features_365days
                            for data_split in target_data_splits]))
    assert np.all(np.array([X_target_365days[data_split].shape[0] == len(Y_target[data_split])
                            for data_split in target_data_splits]))
    
    _, X_target_imputed = predict_missing_features(X_target_30days,
                                                   X_target_30days,
                                                   feature_names_30days,
                                                   important_feature_names,
                                                   X_target_365days,
                                                   X_target_365days,
                                                   feature_names_365days,
                                                   output_dir,
                                                   logger)
    
    logger.info('Evaluating 2020 model on 2020 test data')
    target_pred         = target_model.predict_proba(X_target_30days['test'])[:,1]
    eval_predictions(Y_target['test'],
                     target_pred,
                     logger)
    
    logger.info('Evaluating 2020 model on 2020 imputed test data')
    target_pred_imputed = target_model.predict_proba(X_target_imputed['test'])[:,1]
    eval_predictions(Y_target['test'],
                     target_pred_imputed,
                     logger)
    
def create_parser():
    '''
    Create an argument parser
    @return: argparse ArgumentParser
    '''
    parser = argparse.ArgumentParser(description = ('Predict missing features to handle domain shift.'))
    parser.add_argument('--oracle',
                        action  = 'store_true',
                        default = False,
                        help    = ('Specify whether to learn feature imputation model from 2020 data instead of 2019 data.'))
    return parser
    
if __name__ == '__main__':
    
    parser = create_parser()
    args   = parser.parse_args()
    
    experiment_name     = 'condition_378253_outcomes_from_all_freq100_logreg'
    dataset_name        = 'dataset_condition_378253_outcomes'
    experiment_dir      = config.experiment_dir + experiment_name + '/'
    if args.oracle:
        output_dir      = experiment_dir + 'predict_missing_features_oracle_for_domain_shift_analysis/'
    else:
        output_dir      = experiment_dir + 'predict_missing_features_for_domain_shift_analysis/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dataset_file_header_30days  = config.outcome_data_dir + dataset_name + '/fold0_freq100'
    dataset_file_header_365days = config.outcome_data_dir + dataset_name + '_365day_features/fold0_freq100'
    
    logging_filename = output_dir + experiment_name + '_predict_missing_features_for_domain_shift_' \
                     + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.log'
    logger           = set_up_logger('logger_main',
                                     logging_filename)
    logger.info('python3 predict_missing_features_for_domain_shift.py --oracle=' + str(args.oracle))
    
    source_model = joblib.load(experiment_dir + experiment_name + '_logistic_regression_time2.joblib')
    target_model = joblib.load(experiment_dir + experiment_name + '_logistic_regression_time3.joblib')
    
    Xs_30days, feature_names_30days, _   = load_covariates(dataset_file_header_30days,
                                                           feature_set       = 'all',
                                                           num_years         = 2,
                                                           logger            = logger,
                                                           starting_year_idx = 2)
    X_source_30days                      = {data_split: Xs_30days[data_split][0]
                                            for data_split in Xs_30days}
    X_target_30days                      = {data_split: Xs_30days[data_split][1]
                                            for data_split in Xs_30days}
    
    Xs_365days, feature_names_365days, _ = load_covariates(dataset_file_header_365days,
                                                           feature_set       = 'all',
                                                           num_years         = 2,
                                                           logger            = logger,
                                                           starting_year_idx = 2)
    X_source_365days                     = {data_split: Xs_365days[data_split][0]
                                            for data_split in Xs_365days}
    X_target_365days                     = {data_split: Xs_365days[data_split][1]
                                            for data_split in Xs_365days}
    
    Ys                   = load_outcomes(dataset_file_header_30days,
                                         num_years           = 2,
                                         logger              = logger,
                                         starting_year_idx   = 2)
    Y_source             = {data_split: Ys[data_split][0]
                            for data_split in Ys}
    Y_target             = {data_split: Ys[data_split][1]
                            for data_split in Ys}
    
    important_feature_names   = ['0 - specialty - No matching concept - 30 days',
                                 ('2414398 - procedure - Office or other outpatient visit for the evaluation and '
                                  'management of an established patient, which requires at least 2 of these 3 key components: '
                                  'A detailed history; A detailed examination; Medical decision making of moderate complexity. '
                                  'Counseling and/o - 30 days'),
                                 '433316 - condition - Dizziness and giddiness - 30 days',
                                 ('2414397 - procedure - Office or other outpatient visit for the evaluation and '
                                  'management of an established patient, which requires at least 2 of these 3 key components: '
                                  'An expanded problem focused history; An expanded problem focused examination; '
                                  'Medical decision making of low - 30 days'),
                                 ('2514399 - procedure - Office or other outpatient visit for the evaluation and '
                                  'management of an established patient, which requires at least 2 of these 3 key components: '
                                  'A comprehensive history; A comprehensive examination; '
                                  'Medical decision making of high complexity. Counseling - 30 days'),
                                 '24134 - condition - Neck pain - 30 days',
                                 '38004458 - specialty - Neurology - 30 days',
                                 ('2414393 - procedure - Office or other outpatient visit for the evaluation and '
                                  'management of a new patient, which requires these 3 key components: '
                                  'A comprehensive history; A comprehensive examination; '
                                  'Medical decision making of moderate complexity. Counseling and/or coordinatio - 30 days')]
    
    if args.oracle:
        evaluate_oracle_imputed_features(target_model,
                                         X_target_30days,
                                         Y_target,
                                         feature_names_30days,
                                         important_feature_names,
                                         X_target_365days,
                                         feature_names_365days,
                                         output_dir,
                                         logger)
    else:
        check_for_nonstationarity_with_imputed_features(source_model,
                                                        target_model,
                                                        X_source_30days,
                                                        X_target_30days,
                                                        Y_source,
                                                        Y_target,
                                                        feature_names_30days,
                                                        important_feature_names,
                                                        X_source_365days,
                                                        X_target_365days,
                                                        feature_names_365days,
                                                        output_dir,
                                                        logger)