import sys
import os
import joblib
import time
import json
import gc
import math
from datetime import datetime
import multiprocessing as mp
import multiprocessing_logging
import numpy as np
import pandas as pd
from copy import deepcopy
from itertools import product
from scipy.sparse import csc_matrix
import shutil

from load_data_for_nonstationarity_scan import (
    load_outcomes, 
    load_covariates, 
    load_person_id_to_sample_indices_mappings
)
from set_up_subpopulation_analysis import (
    create_parser, 
    create_config_dict, 
    get_model_errors, 
    get_timepoint_samples
)
from create_regions_for_subpopulation_analysis import (
    create_single_feature_region_indicator_function, 
    create_model_based_region_indicator_function, 
    construct_region_name
)
from handle_age_in_subpopulation_analysis import get_region_age_bound
from learn_and_evaluate_models_over_years import learn_models_over_years
from evaluate_regions_for_subpopulation_analysis import (
    evaluate_models_per_year_on_region,
    check_for_region_nonstationarity
)
from plot_subpopulation_cohort import plot_subpopulation_cohort_and_outcome_frequency

from os.path import dirname, abspath, join
sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from model_utils import eval_predictions
from logging_utils import set_up_logger
from nonstationarity_scan_metric_dict_utils import convert_str_to_num_in_metrics_dict
    
def run_subpopulation_analysis(config_dict,
                               X_csr_all_years_dict,
                               X_csr_all_years_dict_specific,
                               Y_all_years_dict,
                               feature_names,
                               person_id_to_sample_idxs_dict,
                               age_scaled_back,
                               models,
                               thresholds,
                               logger,
                               overwrite = False):
    '''
    Run sub-population analysis to find sub-populations that are more non-stationary
    @param config_dict: dict mapping str to int or str, settings such as file paths, plot titles, number of years
    @param X_csr_all_years_dict: dict mapping str to list of csr matrix, covariates for each year with all features
    @param X_csr_all_years_dict_specific: dict mapping str to list of csr matrix, covariates for each year with only features 
                                          in original logistic regressions
    @param Y_all_years_dict: dict mapping str to list of np arrays, outcomes for each year
    @param feature_names: list of str, covariate names
    @param person_id_to_sample_idxs_dict: dict mapping str to list of defaultdict mapping int to list of ints, 
                                          inner dict contains person ID to sample indices, 
                                          outer dict maps data split to list of inner dicts over time points
    @param age_scaled_back: bool, whether age has been scaled back to normal range in X_csr_all_years_dict
    @param logger: logger, for INFO messages
    @param models: list of sklearn models, one per year
    @param thresholds: list of floats, thresholds for predicted probabilities each year
    @param overwrite: bool, whether to override output files if they already exist
    @return: None
    '''
    if config_dict['baseline']:
        subpopulation_check_filename = config_dict['output_file_header'] \
                                     + 'baseline_nonstationarity_check.csv'
    else:
        subpopulation_check_filename = config_dict['output_file_header'] \
                                     + 'nonstationarity_check.csv'
    if (not overwrite) and os.path.exists(subpopulation_check_filename):
        return
    
    data_splits = ['train', 'valid', 'test']
    assert np.all(np.array([len(X_csr_all_years_dict[data_split]) == config_dict['num_years']
                            for data_split in data_splits]))
    assert np.all(np.array([len(X_csr_all_years_dict_specific[data_split]) == config_dict['num_years']
                            for data_split in data_splits]))
    assert np.all(np.array([len(Y_all_years_dict[data_split]) == config_dict['num_years']
                            for data_split in data_splits]))
    assert np.all(np.array([len(person_id_to_sample_idxs_dict[data_split]) == config_dict['num_years']
                            for data_split in data_splits]))
    for data_split in data_splits:
        assert np.all(np.array([X_csr_all_years_dict[data_split][year_idx].shape[0] 
                                == Y_all_years_dict[data_split][year_idx].shape[0]
                                for year_idx in range(config_dict['num_years'])]))
        assert np.all(np.array([X_csr_all_years_dict_specific[data_split][year_idx].shape[0] 
                                == Y_all_years_dict[data_split][year_idx].shape[0]
                                for year_idx in range(config_dict['num_years'])]))
        assert np.all(np.array([X_csr_all_years_dict[data_split][year_idx].shape[1] == len(feature_names)
                                for year_idx in range(config_dict['num_years'])]))
    assert feature_names[0] == 'age'
    assert {'outcome_name', 'region_identifier', 'get_interaction_terms', 'region_model', 'num_years',
            'experiment_name', 'all_outputs_dir', 'output_file_header', 'analysis_str', 
            'dataset_file_header', 'feature_set'}.issubset(set(config_dict.keys()))
    if config_dict['region_model'] == 'logreg':
        assert not age_scaled_back
    feature_sets_readable = {'cond_proc': 'conditions + procedures',
                             'drugs'    : 'drugs',
                             'labs'     : 'labs',
                             'all'      : 'all features'}
    assert config_dict['feature_set'] in feature_sets_readable
    assert len(models) == config_dict['num_years']
    assert len(thresholds) == config_dict['num_years']
    for year_idx in range(len(models)):
        if models[year_idx] is not None:
            assert thresholds[year_idx] is not None
            for data_split in data_splits:
                assert X_csr_all_years_dict_specific[data_split][year_idx].shape[1] == models[year_idx].n_features_in_
    
    start_time                           = time.time()
    region_Xs_over_years                 = {data_split: [] for data_split in data_splits}
    region_Ys_over_years                 = {data_split: [] for data_split in data_splits}
    if config_dict['region_identifier'] == 'errors':
        test_error_indicators_over_years = []
    else:
        test_error_indicators_over_years = None
    years_with_regions                   = []
    for year_idx in range(1, config_dict['num_years']):
        if config_dict['region_identifier'] == 'errors':
            if models[year_idx - 1] is None or models[year_idx] is None:
                test_error_indicators_over_years.append(None)
                continue
            region_Xs, region_Ys, error_indicators, \
            region_feature_names, region_feature_names_short, region_feature_file_names \
                = get_model_errors(models[year_idx - 1],
                                   models[year_idx],
                                   thresholds[year_idx - 1],
                                   thresholds[year_idx],
                                   {data_split: X_csr_all_years_dict_specific[data_split][year_idx]
                                    for data_split in data_splits},
                                   {data_split: X_csr_all_years_dict[data_split][year_idx]
                                    for data_split in data_splits},
                                   {data_split: Y_all_years_dict[data_split][year_idx]
                                    for data_split in data_splits},
                                   feature_names,
                                   config_dict['outcome_name'],
                                   logger,
                                   get_interaction_terms = config_dict['get_interaction_terms'],
                                   calibration_based     = True)
            test_error_indicators_over_years.append(error_indicators['test'])
            if region_Xs is None:
                continue # not enough test samples with each error label
        else:
            region_Xs, region_Ys, region_feature_names, region_feature_names_short, region_feature_file_names \
                = get_timepoint_samples({data_split: X_csr_all_years_dict[data_split][year_idx-1:year_idx+1]
                                         for data_split in data_splits},
                                        {data_split: Y_all_years_dict[data_split][year_idx-1:year_idx+1]
                                         for data_split in data_splits},
                                        feature_names,
                                        config_dict['outcome_name'],
                                        logger,
                                        get_interaction_terms = config_dict['get_interaction_terms'])
        for data_split in data_splits:
            region_Xs_over_years[data_split].append(region_Xs[data_split])
            region_Ys_over_years[data_split].append(region_Ys[data_split])
        years_with_regions.append(year_idx)
    gc.collect()
    logger.info('Time to set up X and Y for region models: ' + str(time.time() - start_time) + ' seconds')
    df_columns                  = ['Experiment', 'Outcome name', 'Region name', 
                                   '2016', '2017', '2018', '2019', '2020']
    empty_df_data               = {col: [] for col in df_columns}
    nonstationary_df            = pd.DataFrame(data    = empty_df_data,
                                               columns = df_columns)
    if len(years_with_regions) == 0:
        logger.info('No years with enough data for region models')
        nonstationary_df.to_csv(subpopulation_check_filename, index = False)
        return
    
    start_time               = time.time()
    region_model_file_header = config_dict['output_file_header'] + config_dict['analysis_str']
    region_models, region_thresholds, new_region_models_learned \
        = learn_models_over_years(region_Xs_over_years['train'],
                                  region_Ys_over_years['train'],
                                  region_Xs_over_years['valid'],
                                  region_Ys_over_years['valid'],
                                  config_dict['region_model'],
                                  region_feature_names,
                                  region_model_file_header,
                                  config_dict['outcome_name'] + ' from ' + feature_sets_readable[config_dict['feature_set']] 
                                  + ' sub-population',
                                  logger,
                                  years_with_regions,
                                  region_feature_names_short,
                                  overwrite = overwrite)
    if new_region_models_learned:
        overwrite = True
    logger.info('Time to learn region models: ' + str(time.time() - start_time) + ' seconds')
    
    start_time               = time.time()
    for year_idx in range(len(years_with_regions)):
        if region_models[year_idx] is not None:
            test_region_pred = region_models[year_idx].predict_proba(region_Xs_over_years['test'][year_idx])[:,1]
            eval_predictions(region_Ys_over_years['test'][year_idx],
                             test_region_pred,
                             logger,
                             region_thresholds[year_idx])
    logger.info('Time to evaluate region models: ' + str(time.time() - start_time) + ' seconds')
    
    if config_dict['single_feature_regions']:
        prev_region_file_names_dict = set()
    for region_year_idx in range(len(years_with_regions)):
        year_idx = years_with_regions[region_year_idx]
        logger.info('Analyzing year ' + str(year_idx))
        this_start_time = time.time()
        
        for feat_idx in range(5):
            if config_dict['single_feature_regions']:
                assert config_dict['region_model'] == 'logreg'
                coefficients                   = region_models[region_year_idx].coef_.flatten()
                feature_sorted                 = np.argsort(np.abs(coefficients))
                
                this_start_time            = time.time()
                sorted_feat_idx            = feature_sorted[-1*(feat_idx + 1)]
                feature_names_dict         = {'feature_name_full' : region_feature_names[sorted_feat_idx],
                                              'feature_name_short': region_feature_names_short[sorted_feat_idx],
                                              'feature_file_name' : region_feature_file_names[sorted_feat_idx]}
                coefficient                = coefficients[sorted_feat_idx]
                if coefficient == 0:
                    break # feature coefficient is 0, so no more large features to plot
                
                if feature_names_dict['feature_name_short'] == 'age':
                    train_age              = csc_matrix(X_csr_all_years_dict['train'][year_idx])[:,0].toarray()
                    valid_age              = csc_matrix(X_csr_all_years_dict['valid'][year_idx])[:,0].toarray()
                    train_valid_age        = np.concatenate((train_age,
                                                             valid_age))
                    region_age_bound_scaled, region_age_bound_original \
                        = get_region_age_bound(train_valid_age,
                                               coefficient,
                                               config_dict['dataset_file_header'],
                                               age_scaled_back,
                                               logger)
                else:
                    region_age_bound_scaled   = 0
                    region_age_bound_original = 0
                
                region_names_dict          = construct_region_name(feature_names_dict,
                                                                   coefficient,
                                                                   region_age_bound_original)
                logger.info('Sub-population ' + str(feat_idx) + ' in year ' + str(year_idx) 
                            + ' defined as ' + region_names_dict['region_name_full'])
                if config_dict['get_interaction_terms'] \
                    or sorted_feat_idx == region_Xs_over_years['test'][region_year_idx].shape[1] - 1:
                    logger.info('Cannot run non-stationarity check when all samples in region have the same outcome')
                    continue # cannot run non-stationarity check when all samples in region have the same outcome
                if region_names_dict['region_file_name'] in prev_region_file_names_dict:
                    logger.info('Sub-population has been identified before in a previous year.')
                    continue
                prev_region_file_names_dict.add(region_names_dict['region_file_name'])
                
                get_region_indicators = create_single_feature_region_indicator_function(sorted_feat_idx,
                                                                                        coefficient,
                                                                                        config_dict['get_interaction_terms'],
                                                                                        logger,
                                                                                        region_age_bound_scaled)
            else:
                this_start_time       = time.time()
                region_names_dict     = {'region_name_full': 'in year ' + str(year_idx) + ' ' + config_dict['region_model'] 
                                                             + ' region'}
                region_names_dict['region_name_short']     = region_names_dict['region_name_full']
                region_names_dict['region_file_name']      = region_names_dict['region_name_full'].replace(' ', '_')
                region_names_dict['not_region_name_full']  = 'not ' + region_names_dict['region_name_full']
                region_names_dict['not_region_name_short'] = region_names_dict['not_region_name_full']
                region_names_dict['not_region_file_name']  = region_names_dict['not_region_name_full'].replace(' ', '_')
                get_region_indicators = create_model_based_region_indicator_function(region_models[region_year_idx],
                                                                                     region_thresholds[region_year_idx],
                                                                                     config_dict['get_interaction_terms'],
                                                                                     logger)
                
            logger.info('Time to define sub-population ' + str(feat_idx) + ' for year ' + str(year_idx) + ': ' 
                        + str(time.time() - this_start_time) + ' seconds')

            this_start_time            = time.time()
            region_metrics, not_region_metrics, proportion_patients_in_region \
                = evaluate_models_per_year_on_region(config_dict,
                                                     X_csr_all_years_dict['test'],
                                                     X_csr_all_years_dict_specific['test'],
                                                     Y_all_years_dict['test'],
                                                     'test',
                                                     models,
                                                     thresholds,
                                                     region_names_dict,
                                                     get_region_indicators,
                                                     logger,
                                                     errors_all_years = test_error_indicators_over_years,
                                                     overwrite        = overwrite)
            logger.info('Time to evaluate models per year in and not in region ' + str(feat_idx) 
                        + ' for year ' + str(year_idx) + ': ' + str(time.time() - this_start_time) + ' seconds')

            this_start_time = time.time()
            
            region_df       = check_for_region_nonstationarity(config_dict,
                                                               region_names_dict['region_file_name'],
                                                               models,
                                                               get_region_indicators,
                                                               X_csr_all_years_dict['valid'],
                                                               X_csr_all_years_dict_specific['valid'],
                                                               Y_all_years_dict['valid'],
                                                               person_id_to_sample_idxs_dict['valid'],
                                                               X_csr_all_years_dict['test'],
                                                               X_csr_all_years_dict_specific['test'],
                                                               Y_all_years_dict['test'],
                                                               person_id_to_sample_idxs_dict['test'],
                                                               logger,
                                                               overwrite = overwrite,
                                                               only_run_region_def_year = True)
            nonstationary_df = pd.concat((nonstationary_df, region_df))
            logger.info('Time to run non-stationarity check for sub-population ' + str(feat_idx) 
                        + ' in year ' + str(year_idx) + ': ' + str(time.time() - this_start_time) + ' seconds')
            
            this_start_time = time.time()
            plot_subpopulation_cohort_and_outcome_frequency(config_dict,
                                                            X_csr_all_years_dict,
                                                            Y_all_years_dict,
                                                            region_names_dict,
                                                            get_region_indicators,
                                                            logger,
                                                            overwrite = overwrite)
            logger.info('Time to plot cohort for sub-population ' + str(feat_idx)
                        + ' in year ' + str(year_idx) + ': ' + str(time.time() - this_start_time) + ' seconds')
            
            if not config_dict['single_feature_regions']:
                break # only one region per year
    
    nonstationary_df.to_csv(subpopulation_check_filename, 
                            index = False)
    logger.info('Saved sub-population hypothesis testing results to ' + subpopulation_check_filename)
    
if __name__ == '__main__':
    
    mp.set_start_method('spawn', force=True)
    
    start_time = time.time()
    parser = create_parser()
    args = parser.parse_args()
    assert args.outcome  in {'eol', 'condition', 'procedure', 'lab'}
    feature_sets_readable = {'cond_proc': 'conditions + procedures',
                             'drugs'    : 'drugs',
                             'labs'     : 'labs',
                             'all'      : 'all features'}
    assert args.features in feature_sets_readable
    if args.outcome in {'condition', 'procedure', 'lab'}:
        assert len(args.outcome_id) > 0
        assert ' ' not in args.outcome_id
        assert len(args.outcome_name) > 0
        if args.outcome == 'lab':
            assert args.direction in {'low', 'high'}
    assert args.model in {'logreg', 'dectree', 'forest', 'xgboost'}
    assert args.region_model in {'logreg', 'dectree', 'forest', 'xgboost'}
    if args.region_model != 'logreg':
        assert not args.interactions
    assert args.region_identifier in {'errors', 'timepoint'}
    assert args.fold in {0, 1, 2, 3}
    
    config_dict = create_config_dict(args)
    if not os.path.exists(config_dict['output_dir']):
        os.makedirs(config_dict['output_dir'])
    time_str         = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    logging_filename = config_dict['output_file_header'] + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.log'
    logger           = set_up_logger('logger_main',
                                     logging_filename)
    multiprocessing_logging.install_mp_handler()
    logger.info('python3 run_subpopulation_analysis.py'
                + ' --outcome='                 + args.outcome 
                + ' --outcome_id='              + str(args.outcome_id) 
                + ' --direction='               + args.direction
                + ' --features='                + args.features
                + ' --outcome_name='            + args.outcome_name
                + ' --model='                   + args.model
                + ' --region_model='            + args.region_model
                + ' --region_identifier='       + args.region_identifier
                + ' --single_feature_regions='  + str(args.single_feature_regions)
                + ' --interactions='            + str(args.interactions)
                + ' --debug_size='              + str(args.debug_size)
                + ' --baseline='                + str(args.baseline)
                + ' --feature_windows='         + str(args.feature_windows)
                + ' --fold='                    + str(args.fold))
    
    if config_dict['baseline']:
        summary_filename = config_dict['output_file_header'] \
                         + 'subpopulation_baseline_nonstationarity_check.csv'
    else:
        summary_filename = config_dict['output_file_header'] \
                         + 'subpopulation_nonstationarity_check.csv'
    if os.path.exists(summary_filename):
        logger.info('Sub-population analysis has been run before. Non-stationary regions are at ' + summary_filename)
        sys.exit()
    
    outcome_start_time  = time.time()
    Y_all_years_dict    = load_outcomes(config_dict['dataset_file_header'],
                                        config_dict['num_years'],
                                        logger,
                                        test_dataset_file_header = config_dict['test_data_file_header'])
    logger.info('Time to load outcomes: ' + str(time.time() - outcome_start_time) + ' seconds')

    covariate_start_time = time.time()
    scale_age_back       = False
    if config_dict['region_model'] != 'logreg':
        scale_age_back   = True # scale before passing into dectree or forest for better interpretation
    X_csr_all_years_dict, feature_names, age_scaled_back \
        = load_covariates(config_dict['dataset_file_header'],
                          'all',
                          config_dict['num_years'],
                          logger,
                          scale_age_back           = scale_age_back,
                          test_dataset_file_header = config_dict['test_data_file_header'])
    X_csr_all_years_dicts = [X_csr_all_years_dict]
    logger.info('Time to load all covariates: ' + str(time.time() - covariate_start_time) + ' seconds')
    
    if config_dict['feature_set'] != 'all' or age_scaled_back:
        specific_start_time = time.time()
        X_csr_all_years_dict_specific, specific_feature_names, _ \
            = load_covariates(config_dict['dataset_file_header'],
                              config_dict['feature_set'],
                              config_dict['num_years'],
                              logger,
                              test_dataset_file_header = config_dict['test_data_file_header'])
        X_csr_all_years_dicts.append(X_csr_all_years_dict_specific)
        logger.info('Time to load specific covariates: ' + str(time.time() - specific_start_time) + ' seconds')
    else:
        specific_feature_names = feature_names
         
    this_start_time = time.time()
    person_id_to_sample_idxs_dict \
        = load_person_id_to_sample_indices_mappings(config_dict['dataset_file_header'],
                                                    config_dict['num_years'],
                                                    starting_year_idx        = config_dict['starting_year_idx'],
                                                    test_dataset_file_header = config_dict['test_data_file_header'])
    logger.info('Time to load person ID to sample indices mappings: ' + str(time.time() - this_start_time) + ' seconds')
            
    X_csr_all_years_dict          = X_csr_all_years_dicts[0]
    X_csr_all_years_dict_specific = X_csr_all_years_dicts[-1]
    
    if args.region_identifier == 'errors':
        model_start_time   = time.time()
        models, thresholds, new_models_learned = learn_models_over_years(X_csr_all_years_dict_specific['train'], 
                                                                         Y_all_years_dict['train'], 
                                                                         X_csr_all_years_dict_specific['valid'],
                                                                         Y_all_years_dict['valid'],
                                                                         config_dict['model'],
                                                                         specific_feature_names,
                                                                         config_dict['orig_output_file_header'],
                                                                         config_dict['outcome_name'] + ' from ' 
                                                                         + feature_sets_readable[config_dict['feature_set']],
                                                                         logger)
        logger.info('Time to load or learn models per year: ' + str(time.time() - model_start_time) + ' seconds')
    else:
        models    = None
        thresholds = None
        
    subpopulation_start_time = time.time()
    run_subpopulation_analysis(config_dict,
                               X_csr_all_years_dict,
                               X_csr_all_years_dict_specific,
                               Y_all_years_dict,
                               feature_names,
                               person_id_to_sample_idxs_dict,
                               age_scaled_back,
                               models,
                               thresholds,
                               logger,
                               overwrite = new_models_learned)
    logger.info('Time to run sub-population analysis: ' + str(time.time() - subpopulation_start_time) + ' seconds')
    
    logger.info('Total time to run analysis: ' + str(time.time() - start_time) + ' seconds')