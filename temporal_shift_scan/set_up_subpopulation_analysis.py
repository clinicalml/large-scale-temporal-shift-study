import argparse
import numpy as np
import sys
import os
from scipy.sparse import hstack, vstack
from os.path import dirname, abspath, join

from create_interactions_and_feature_names import (
    create_interaction_terms, 
    get_all_feature_names_without_interactions
)

sys.path.append(dirname(dirname(abspath(__file__))))
import config

def create_parser():
    '''
    Create an argument parser
    @return: argparse ArgumentParser
    '''
    parser = argparse.ArgumentParser(description='Analyze errors due to dataset shift and sub-populations.')
    parser.add_argument('--outcome', 
                        action  = 'store', 
                        type    = str, 
                        help    = 'Specify outcome among eol, condition, procedure, lab, lab_group')
    parser.add_argument('--outcome_id', 
                        action  = 'store', 
                        type    = str, 
                        default = '', 
                        help    = ('Specify ID of condition or lab outcome or string for procedure. '
                                   'No spaces allowed since used for file names.')
                       )
    parser.add_argument('--outcome_ids',
                        action  = 'store',
                        type    = str,
                        default = '',
                        help    = 'Specify comma-separated list of lab outcome IDs for lab group outcomes.')
    parser.add_argument('--direction',
                        action  = 'store',
                        type    = str,
                        default = '',
                        help    = 'Specify low or high for abnormal lab outcome')
    parser.add_argument('--features', 
                        action  = 'store', 
                        type    = str, 
                        help    = 'Specify features among cond_proc, drugs, labs, all')
    parser.add_argument('--outcome_name',
                        action  = 'store',
                        type    = str,
                        default = '',
                        help    = 'Specify name of condition, procedure, or lab outcome for plot title.')
    parser.add_argument('--model',
                        action  = 'store',
                        type    = str,
                        help    = 'Specify type of original model: logreg, dectree, forest, or xgboost.')
    parser.add_argument('--region_model',
                        action  = 'store',
                        type    = str,
                        help    = 'Specify type of region model: logreg, dectree, forest, or xgboost.')
    parser.add_argument('--region_identifier',
                        action  = 'store',
                        type    = str,
                        help    = 'Specify set-up used to identify model: errors or timepoint.')
    parser.add_argument('--single_feature_regions',
                        action  = 'store_true',
                        default = False,
                        help    = ('Specify whether regions should be defined by single features '
                                   'based on logistic regression coefficients.')
                       )
    parser.add_argument('--interactions',
                        action  = 'store_true',
                        default = False,
                        help    = 'Specify whether to create interaction terms between X and Y.')
    parser.add_argument('--debug_size',
                        action  = 'store',
                        type    = int,
                        default = None,
                        help    = 'Specify smaller cohort size for debugging.')
    parser.add_argument('--baseline',
                        action  = 'store_true',
                        default = False,
                        help    = 'Specify whether to perform baseline evaluation instead.')
    parser.add_argument('--feature_windows',
                        action  = 'store',
                        type    = str,
                        default = '30',
                        help    = 'Specify comma-separated list of feature window lengths in days.')
    parser.add_argument('--fold',
                        action  = 'store',
                        type    = int,
                        default = 0,
                        help    = ('Specify which fold to use data from: 0, 1, 2, or 3.'))
    return parser

def get_arg(args,
            remaining_args,
            arg_name):
    '''
    Check arg_name is only in args or remaining_args and return its value
    @param args: arguments from parser
    @param remaining_args: dict, str mapping argument name to value if not present in args
    @param arg_name: str, name of argument
    @return: int or str, value for arg_name
    '''
    if arg_name in vars(args):
        assert arg_name not in remaining_args
        return getattr(args, arg_name)
    assert arg_name in remaining_args
    return remaining_args[arg_name]

def create_config_dict(args,
                       remaining_args = dict()):
    '''
    Create configuration dictionary from arguments
    @param args: arguments from parser
    @param remaining_args: dict, str mapping argument name to value if not present in args, 
                           only for region_model, region_identifier, single_feature_regions, and interactions
    @return: dict mapping str to ints and strs specifying file paths, plot titles, number of years, and other settings
    '''
    
    config_dict                      = dict()
    config_dict['outcome']           = args.outcome 
    config_dict['outcome_id']        = args.outcome_id
    config_dict['outcome_ids']       = args.outcome_ids
    config_dict['direction']         = args.direction
    config_dict['feature_set']       = args.features
    config_dict['outcome_name']      = args.outcome_name
    config_dict['model']             = args.model
    config_dict['baseline']          = args.baseline
    config_dict['fold']              = args.fold
    config_dict['feature_windows']   = [int(days) for days in args.feature_windows.split(',')]
    if len(config_dict['feature_windows']) == 1 and config_dict['feature_windows'][0] == 30:
        config_dict['feature_window_suffix'] = ''
    else:
        config_dict['feature_window_suffix'] = '_' + '_'.join(map(str, config_dict['feature_windows'])) + 'day_features'
    config_dict['region_model']      = get_arg(args,
                                               remaining_args,
                                               'region_model')
    assert config_dict['region_model'] in {'logreg', 'dectree', 'forest', 'xgboost'}
    config_dict['region_identifier'] = get_arg(args,
                                               remaining_args,
                                               'region_identifier')
    config_dict['single_feature_regions'] = get_arg(args,
                                                    remaining_args,
                                                    'single_feature_regions')
    if config_dict['region_model'] != 'logreg':
        assert not config_dict['single_feature_regions']
    if args.debug_size is not None:
        config_dict['debug_suffix']  = '_debug' + str(args.debug_size)
    else:
        config_dict['debug_suffix']  = ''
    
    config_dict['data_dir']     = config.outcome_data_dir
    config_dict['tmp_data_dir'] = config.interaction_dir
    feature_plot_titles         = {'cond_proc': 'conditions and procedures',
                                   'drugs'    : 'drugs',
                                   'labs'     : 'labs',
                                   'all'      : 'all features'}
    if args.outcome == 'condition':
        # account for smaller cohort size
        config_dict['min_freq']         = 100
        config_dict['num_years']        = 4 # years: 2017 - 2020
        config_dict['starting_year']    = 2017
        config_dict['eligibility_time'] = '3 years'
    else:
        config_dict['min_freq']         = 300
        config_dict['num_years']        = 6 # years: 2015 - 2020
        config_dict['starting_year']    = 2015
        config_dict['eligibility_time'] = '1 year'
    config_dict['starting_year_idx']    = 0
    config_dict['all_outputs_dir']      = config.experiment_dir
    if args.outcome == 'eol':
        config_dict['outcome_specific_data_dir'] = config_dict['data_dir'] + 'dataset_eol_outcomes' \
                                                 + config_dict['feature_window_suffix'] + config_dict['debug_suffix'] + '/'
        config_dict['experiment_name']       = 'eol_outcomes_from_' + args.features + '_freq' + str(config_dict['min_freq']) \
                                             + '_' + config_dict['model'] + config_dict['feature_window_suffix'] \
                                             + config_dict['debug_suffix']
        config_dict['plot_title']            = 'Mortality from ' + feature_plot_titles[args.features]
        config_dict['outcome_as_feat']       = 'Death'
        config_dict['cohort_plot_header']    = 'eol_outcomes' + config_dict['debug_suffix']
    elif args.outcome in {'condition', 'procedure'}:
        config_dict['outcome_specific_data_dir'] = config_dict['data_dir'] + 'dataset_' + args.outcome + '_' + args.outcome_id \
                                                 + '_outcomes' + config_dict['feature_window_suffix'] \
                                                 + config_dict['debug_suffix'] + '/'
        config_dict['experiment_name']       = args.outcome + '_' + args.outcome_id + '_outcomes_from_' + args.features \
                                             + '_freq' + str(config_dict['min_freq']) + '_' + config_dict['model'] \
                                             + config_dict['feature_window_suffix'] + config_dict['debug_suffix']
        config_dict['plot_title']            = args.outcome_name + ' from ' + feature_plot_titles[args.features]
        config_dict['outcome_as_feat']       = args.outcome_name
        config_dict['cohort_plot_header']    = args.outcome + '_' + args.outcome_id + '_outcomes' + config_dict['debug_suffix']
    else:
        config_dict['outcome_specific_data_dir'] = config_dict['data_dir'] + 'dataset_' + args.outcome + '_' + args.outcome_id \
                                                 + '_' + args.direction + '_outcomes' + config_dict['feature_window_suffix'] \
                                                 + config_dict['debug_suffix'] + '/'
        config_dict['experiment_name']       = args.outcome + '_' + args.outcome_id + '_' + args.direction \
                                             + '_outcomes_from_' + args.features + '_freq' + str(config_dict['min_freq']) \
                                             + '_' + config_dict['model'] + config_dict['feature_window_suffix'] \
                                             + config_dict['debug_suffix']
        config_dict['plot_title']            = args.outcome_name + ' ' + args.direction \
                                             +  ' from ' + feature_plot_titles[args.features]
        config_dict['outcome_as_feat']       = args.direction + ' ' + args.outcome_name
        config_dict['cohort_plot_header']    = args.outcome + '_' + args.outcome_id + '_outcomes' + config_dict['debug_suffix']
    
    if args.fold == 0:
        fold_str                             = ''
        data_fold_str                        = 'fold' + str(args.fold)
        test_fold_str                        = 'fold0'
    else:
        fold_str                             = '_fold' + str(args.fold) + '_using_fold0_features'
        data_fold_str                        = 'fold' + str(args.fold) + '_using_fold0'
        test_fold_str                        = 'fold0'
    config_dict['dataset_file_header']       = config_dict['outcome_specific_data_dir'] + data_fold_str + '_freq' \
                                             + str(config_dict['min_freq'])
    config_dict['test_data_file_header']     = config_dict['outcome_specific_data_dir'] + test_fold_str + '_freq' \
                                             + str(config_dict['min_freq'])
    config_dict['get_interaction_terms']     = get_arg(args,
                                                       remaining_args,
                                                       'interactions')
    config_dict['analysis_str']              = 'subpopulation_analysis_' + config_dict['region_identifier'] + '_' \
                                             + config_dict['region_model'] + '_'
    if config_dict['get_interaction_terms']:
        config_dict['analysis_str']         += 'with_interact_'
    config_dict['output_dir']                = config_dict['all_outputs_dir'] + config_dict['experiment_name'] + '/' \
                                             + config_dict['analysis_str'][:-1] + fold_str + '/'
    if not os.path.exists(config_dict['output_dir']):
        os.makedirs(config_dict['output_dir'])
    config_dict['orig_output_file_header']   = config_dict['all_outputs_dir'] + config_dict['experiment_name'] + fold_str + '/' \
                                             + config_dict['experiment_name'] + '_'
    config_dict['output_file_header']        = config_dict['output_dir'] + config_dict['experiment_name'] + '_' \
                                             + config_dict['analysis_str']
    return config_dict

def get_model_errors(prev_model,
                     curr_model,
                     prev_threshold,
                     curr_threshold,
                     X_specific,
                     X_all,
                     Y,
                     feature_names,
                     outcome_name,
                     logger,
                     valid_split_size      = .25,
                     min_samples           = 10,
                     get_interaction_terms = False,
                     calibration_based     = False):
    '''
    Identify samples that are predicted correctly by curr_model and label whether they are predicted correctly by prev_model.
    If calibration-based, identifies samples where predicted probability from curr_model is closer to true label
    than prev_model.
    Because train/valid error distributions may differ, they are re-split stratified by error label for this analysis.
    If there are fewer than min_samples with either error label in any split, logs and returns Nones
    Features will contain X and Y. Separate features will be created for (X, Y=0) and (X, Y=1) if getting interaction terms.
    @param prev_model: sklearn LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, or xgboost XGBClassifier
                       from previous year
    @param curr_model: sklearn LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, or xgboost XGBClassifier
                       from current year
    @param prev_threshold: float, threshold for prev_model predicted probabilities
    @param curr_threshold: float, threshold for curr_model predicted probabilities
    @param X_specific: dict, str to csr matrix, data split to covariates from current year, feature set in models
    @param X_all: dict, str to csr matrix, data split to covariates from current year, 
                  entire feature set used for downstream error models, 
                  matches sample order in X_specific for each data split
    @param Y: dict, str to np array, data split to outcomes from current year,
              matches sample order in X_specific for each data split
    @param feature_names: list of str, names of features in X_all
    @param outcome_name: str, name of outcome in Y
    @param logger: logger, for INFO messages
    @param valid_split_size: float, 0 to 1, proportion of data used for validation in each outcome stratification
    @param min_samples: int, minimum number of samples with either error label in each split
    @param get_interaction_terms: bool, whether to get interaction terms
    @param calibration_based: bool, whether to use probabilities instead of thresholded binary predictions
    @return: 1. error_Xs: dict, str to csr matrix, data split to X_all samples that are predicted correctly by curr_model,
                          contains all samples if calibration-based
             2. error_Ys: dict, str to np array, data split to binary indicators for whether samples in error_Xs
                          are predicted correctly by prev_model,
                          indicators for whether prev_model predictions are at least as close to true label as curr_model
                          if calibration-based
             3. orig_errors: dict, str to np array, data split to binary indicators for samples that are predicted 
                             correctly by curr_model and incorrectly by prev_model, 
                             differs from error_Ys in that this matches the samples in the original dataset
             3. list of str, feature names corresponding to error_Xs (with interactions if present)
             4. list of str, shortened feature names corresponding to error_Xs (with interactions if present)
             5. list of str, feature file names corresponding to error_Xs (with interactions if present)
    '''
    data_splits = ['train', 'valid', 'test']
    assert set(data_splits).issubset(X_specific.keys())
    assert set(data_splits).issubset(X_all.keys())
    assert set(data_splits).issubset(Y.keys())
    num_specific_feats = prev_model.n_features_in_
    assert curr_model.n_features_in_ == num_specific_feats
    assert np.all(np.array([X_specific[data_split].shape[1] == num_specific_feats
                            for data_split in data_splits]))
    assert np.all(np.array([X_all[data_split].shape[0] == X_specific[data_split].shape[0]
                            for data_split in data_splits]))
    assert np.all(np.array([Y[data_split].shape[0] == X_specific[data_split].shape[0]
                            for data_split in data_splits]))
    num_all_feats = len(feature_names)
    assert np.all(np.array([X_all[data_split].shape[1] == num_all_feats
                            for data_split in data_splits]))
    assert valid_split_size > 0 and valid_split_size < 1
    assert min_samples > 0
    
    '''
    c1_p1: correctly predicted by curr_model and prev_model,
           prev_model prediction at least as close to true label as curr_model prediction if calibration-based
    c1_p0: correctly predicted by curr_model and incorrectly predicted by prev_model
           prev_model prediction farther from true label than curr_model prediction if calibration-based
    '''
    c1_p1_indices                        = dict()
    c1_p0_indices                        = dict()
    c1_p0_indicators                     = dict()
    if calibration_based:
        for data_split in data_splits:
            curr_pred                    = curr_model.predict_proba(X_specific[data_split])[:,1]
            curr_pred_distance           = np.abs(curr_pred - Y[data_split])
            prev_pred                    = prev_model.predict_proba(X_specific[data_split])[:,1]
            prev_pred_distance           = np.abs(prev_pred - Y[data_split])
            c1_p1_indices[data_split]    = np.nonzero(np.where(prev_pred_distance <= curr_pred_distance, 1, 0))[0]
            num_c1_p1                    = len(c1_p1_indices[data_split])
            c1_p0_indicators[data_split] = np.where(prev_pred_distance > curr_pred_distance, 1, 0)
            c1_p0_indices[data_split]    = np.nonzero(c1_p0_indicators[data_split])[0]
            num_c1_p0                    = len(c1_p0_indices[data_split])
            num_samples                  = len(Y[data_split])
            logger.info(data_split + ' model at t - 1 calibrated at least as well as model at t: ' 
                        + str(num_c1_p1) + ' samples, proportion: ' + str(num_c1_p1/float(num_samples)))
            logger.info(data_split + ' model at t - 1 not as well calibrated as model at t: ' 
                        + str(num_c1_p0) + ' samples, proportion: ' + str(num_c1_p0/float(num_samples)))
    else:
        for data_split in data_splits:
            curr_pred                    = np.where(curr_model.predict_proba(X_specific[data_split])[:,1] >= curr_threshold, 
                                                    1, 0)
            curr_correct_indicators      = np.where(curr_pred == Y[data_split], 1, 0)
            prev_pred                    = np.where(prev_model.predict_proba(X_specific[data_split])[:,1] >= prev_threshold, 
                                                    1, 0)
            prev_correct_indicators      = np.where(prev_pred == Y[data_split], 1, 0)
            c1_p1_indices[data_split]    = np.nonzero(np.where(np.logical_and(curr_correct_indicators == 1,
                                                                              prev_correct_indicators == 1),
                                                               1, 0))[0]
            num_c1_p1                    = len(c1_p1_indices[data_split])
            c1_p0_indicators[data_split] = np.where(np.logical_and(curr_correct_indicators == 1,
                                                                   prev_correct_indicators == 0),
                                                    1, 0)
            c1_p0_indices[data_split]    = np.nonzero(c1_p0_indicators[data_split])[0]
            num_c1_p0                    = len(c1_p0_indices[data_split])
            num_c0_p1                    = np.sum(np.where(np.logical_and(curr_correct_indicators == 0,
                                                                          prev_correct_indicators == 1),
                                                           1, 0))
            num_c0_p0                    = np.sum(np.where(np.logical_and(curr_correct_indicators == 0,
                                                                          prev_correct_indicators == 0),
                                                           1, 0))
            num_samples                  = len(Y[data_split])
            logger.info(data_split + ' model at t correct, model at t-1 correct: ' 
                        + str(num_c1_p1) + ' samples, proportion: ' + str(num_c1_p1/float(num_samples)))
            logger.info(data_split + ' model at t correct, model at t-1 incorrect: ' 
                        + str(num_c1_p0) + ' samples, proportion: ' + str(num_c1_p0/float(num_samples)))
            logger.info(data_split + ' model at t incorrect, model at t-1 correct: ' 
                        + str(num_c0_p1) + ' samples, proportion: ' + str(num_c0_p1/float(num_samples)))
            logger.info(data_split + ' model at t incorrect, model at t-1 incorrect: ' 
                        + str(num_c0_p0) + ' samples, proportion: ' + str(num_c0_p0/float(num_samples)))
    
    if len(c1_p1_indices['test']) < min_samples or len(c1_p0_indices['test']) < min_samples:
        logger.info('Not enough test samples with each error label.')
        return None, None, c1_p0_indicators, None, None, None
    
    if len(c1_p1_indices['train']) + len(c1_p1_indices['valid']) < min_samples/valid_split_size \
        or len(c1_p0_indices['train']) + len(c1_p0_indices['valid']) < min_samples/valid_split_size:
        logger.info('Not enough train and valid samples with each error label.')
        return None, None, c1_p0_indicators, None, None, None
    
    test_c1_indices = np.concatenate((c1_p1_indices['test'], 
                                      c1_p0_indices['test']))
    test_c1_labels  = np.concatenate((np.zeros(len(c1_p1_indices['test'])), 
                                      np.ones(len(c1_p0_indices['test']))))
    test_idxs       = np.arange(len(test_c1_labels))
    np.random.seed(1027)
    np.random.shuffle(test_idxs)
    if get_interaction_terms:
        test_error_Xs, error_feature_names, error_feature_names_short, error_feature_file_names \
            = create_interaction_terms(X_all['test'][test_c1_indices[test_idxs]],
                                       Y['test'][test_c1_indices[test_idxs]],
                                       logger,
                                       get_feature_names_with_interactions = True,
                                       feature_names                       = feature_names,
                                       outcome_name                        = outcome_name)
        error_Xs = {'test': hstack((test_error_Xs,
                                    np.expand_dims(Y['test'][test_c1_indices[test_idxs]], axis=1)),
                                   format='csr')}
    else:
        error_Xs = {'test': hstack((X_all['test'][test_c1_indices[test_idxs]],
                                    np.expand_dims(Y['test'][test_c1_indices[test_idxs]], axis=1)),
                                   format='csr')}
        error_feature_names, error_feature_names_short, error_feature_file_names \
            = get_all_feature_names_without_interactions(feature_names)
    error_feature_names.append(outcome_name + ' (Y)')
    error_feature_names_short.append(outcome_name[:min(len(outcome_name), 25)] + ' (Y)')
    error_feature_file_names.append(outcome_name.replace(' ', '_') + '_Y')
    error_Ys     = {'test': test_c1_labels[test_idxs]}
    
    train_valid_c1_p1_indices = np.concatenate((c1_p1_indices['train'], 
                                                X_all['train'].shape[0] + c1_p1_indices['valid']))
    np.random.shuffle(train_valid_c1_p1_indices)
    train_valid_c1_p0_indices = np.concatenate((c1_p0_indices['train'], 
                                                X_all['train'].shape[0] + c1_p0_indices['valid']))
    np.random.shuffle(train_valid_c1_p0_indices)
    new_train_c1_p1_indices   = train_valid_c1_p1_indices[int(valid_split_size * len(train_valid_c1_p1_indices)):]
    new_valid_c1_p1_indices   = train_valid_c1_p1_indices[:int(valid_split_size * len(train_valid_c1_p1_indices))]
    new_train_c1_p0_indices   = train_valid_c1_p0_indices[int(valid_split_size * len(train_valid_c1_p0_indices)):]
    new_valid_c1_p0_indices   = train_valid_c1_p0_indices[:int(valid_split_size * len(train_valid_c1_p0_indices))]
    new_train_indices         = np.concatenate((new_train_c1_p1_indices, 
                                                new_train_c1_p0_indices))
    new_train_labels          = np.concatenate((np.zeros(len(new_train_c1_p1_indices)), 
                                                np.ones(len(new_train_c1_p0_indices))))
    new_valid_indices         = np.concatenate((new_valid_c1_p1_indices,
                                                new_valid_c1_p0_indices))
    new_valid_labels          = np.concatenate((np.zeros(len(new_valid_c1_p1_indices)),
                                                np.ones(len(new_valid_c1_p0_indices))))
    new_train_idxs            = np.arange(len(new_train_labels))
    new_valid_idxs            = np.arange(len(new_valid_labels))
    np.random.shuffle(new_train_idxs)
    np.random.shuffle(new_valid_idxs)
    
    X_all_train_valid         = vstack((X_all['train'],
                                        X_all['valid']),
                                       format='csr')
    Y_all_train_valid         = np.concatenate((Y['train'],
                                                Y['valid']))
    if get_interaction_terms:
        train_valid_feats     = hstack((create_interaction_terms(X_all_train_valid,
                                                                 Y_all_train_valid,
                                                                 logger),
                                        np.expand_dims(Y_all_train_valid, axis=1)),
                                       format='csr')
    else:
        train_valid_feats     = hstack((X_all_train_valid,
                                        np.expand_dims(Y_all_train_valid, axis=1)),
                                       format='csr')
    error_Xs['train']         = train_valid_feats[new_train_indices[new_train_idxs]]
    error_Xs['valid']         = train_valid_feats[new_valid_indices[new_valid_idxs]]
    error_Ys['train']         = new_train_labels[new_train_idxs]
    error_Ys['valid']         = new_valid_labels[new_valid_idxs]
    
    return error_Xs, error_Ys, c1_p0_indicators, error_feature_names, error_feature_names_short, error_feature_file_names

def get_timepoint_samples(Xs,
                          Ys,
                          feature_names,
                          outcome_name,
                          logger,
                          valid_split_size      = .25,
                          get_interaction_terms = False):
    '''
    Create a dataset to predict which timepoint each sample came from.
    Because train/valid error distributions may differ, they are re-split stratified by error label for this analysis.
    Features will contain X and Y. Separate features will be created for (X, Y=0) and (X, Y=1) if getting interaction terms.
    @param Xs: dict, str to list of 2 csr matrices, data split to features for two years
    @param Ys: dict, str to list of 2 np arrays, data split to outcomes for two years
    @param feature_names: list of str, names of features in X
    @param outcome_name: str, name of outcome in Y
    @param logger: logger, for INFO messages
    @param valid_split_size: float, 0 to 1, proportion of data used for validation in each outcome stratification
    @param get_interaction_terms: bool, whether to get interaction terms
    @return: 1. timepoint_Xs: dict, str to csr matrix, data split to samples from both time points with X, Y as features
             2. timepoint_Ys: dict, str to np array, data split to binary indicators for which time point samples came from
             3. list of str, feature names corresponding to timepoint_Xs (with interactions if present)
             4. list of str, shortened feature names corresponding to timepoint_Xs (with interactions if present)
             5. list of str, feature file names corresponding to timepoint_Xs (with interactions if present)
    '''
    data_splits    = ['train', 'valid', 'test']
    assert set(data_splits).issubset(Xs.keys())
    assert set(data_splits).issubset(Ys.keys())
    num_time_steps = 2
    assert np.all(np.array([len(Xs[data_split]) == num_time_steps
                            for data_split in data_splits]))
    assert np.all(np.array([len(Ys[data_split]) == num_time_steps
                            for data_split in data_splits]))
    num_feats      = len(feature_names)
    for data_split in data_splits:
        assert np.all(np.array([Xs[data_split][time_idx].shape[1] == num_feats
                                for time_idx in range(num_time_steps)]))
    assert valid_split_size > 0 and valid_split_size < 1
    
    timepoint_Xs             = dict()
    timepoint_Ys             = dict()
    both_train_valid_Xs      = vstack((Xs['train'][0],
                                       Xs['train'][1],
                                       Xs['valid'][0],
                                       Xs['valid'][1]),
                                      format='csr')
    both_train_valid_Ys      = np.concatenate((Ys['train'][0],
                                               Ys['train'][1],
                                               Ys['valid'][0],
                                               Ys['valid'][1]))
    both_test_Xs             = vstack((Xs['test'][0],
                                       Xs['test'][1]),
                                      format='csr')
    both_test_Ys             = np.concatenate((Ys['test'][0],
                                               Ys['test'][1]))
    if get_interaction_terms:
        train_valid_feats    = hstack((create_interaction_terms(both_train_valid_Xs,
                                                                both_train_valid_Ys,
                                                                logger),
                                       np.expand_dims(both_train_valid_Ys, axis=1)),
                                      format='csr')
        test_timepoint_Xs, timepoint_feature_names, timepoint_feature_names_short, timepoint_feature_file_names \
            = create_interaction_terms(both_test_Xs,
                                       both_test_Ys,
                                       logger,
                                       get_feature_names_with_interactions = True,
                                       feature_names                       = feature_names,
                                       outcome_name                        = outcome_name)
        timepoint_Xs['test'] = hstack((test_timepoint_Xs,
                                       np.expand_dims(both_test_Ys, axis=1)),
                                      format='csr')
    else:
        train_valid_feats    = hstack((both_train_valid_Xs,
                                       np.expand_dims(both_train_valid_Ys, axis=1)),
                                      format='csr')
        timepoint_Xs['test'] = hstack((both_test_Xs,
                                       np.expand_dims(both_test_Ys, axis=1)),
                                      format='csr')
        timepoint_feature_names, timepoint_feature_names_short, timepoint_feature_file_names \
            = get_all_feature_names_without_interactions(feature_names)
    train_valid_labels       = np.concatenate((np.zeros(len(Ys['train'][0])),
                                               np.ones(len(Ys['train'][1])),
                                               np.zeros(len(Ys['valid'][0])),
                                               np.ones(len(Ys['valid'][1]))))
    timepoint_Ys['test']     = np.concatenate((np.zeros(len(Ys['test'][0])),
                                               np.ones(len(Ys['test'][1]))))
    np.random.seed(1027)
    test_idxs = np.arange(len(timepoint_Ys['test']))
    np.random.shuffle(test_idxs)
    timepoint_Xs['test'] = timepoint_Xs['test'][test_idxs]
    timepoint_Ys['test'] = timepoint_Ys['test'][test_idxs]
    
    timepoint_Xs['train'], timepoint_Xs['valid'], timepoint_Ys['train'], timepoint_Ys['valid'] \
        = train_test_split(train_valid_feats,
                           train_valid_labels,
                           test_size    = valid_split_size,
                           random_state = 1031)
    
    return timepoint_Xs, timepoint_Ys, timepoint_feature_names, timepoint_feature_names_short, timepoint_feature_file_names