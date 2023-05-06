import numpy as np
import pandas as pd
from scipy.sparse import vstack, hstack, csr_matrix
import json
import matplotlib.pyplot as plt
from itertools import product
import sys
import os
import time
from datetime import datetime
import multiprocessing as mp
import multiprocessing_logging

from load_data_for_nonstationarity_scan import (
    load_outcomes, 
    load_covariates, 
    load_person_id_to_sample_indices_mappings
)
from create_interactions_and_feature_names import get_all_feature_names_without_interactions
from learn_and_evaluate_models_over_years import (
    learn_models_over_years,
    evaluate_models_on_future_years
)
from run_subpopulation_analysis import run_subpopulation_analysis
from set_up_nonstationarity_check import create_parser, create_config_dict
from set_up_subpopulation_analysis import create_config_dict as create_subpopulation_config_dict

from os.path import dirname, abspath, join
sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from logging_utils import set_up_logger
from nonstationarity_scan_metric_dict_utils import convert_str_to_num_in_metrics_dict
from nonstationarity_check_utils import (
    satisfy_patient_outcome_count_minimum,
    satisfy_auc_minimum,
    satisfy_auc_diff_2_models_1_dataset_patient_bootstrap_ci_above_0,
    satisfy_auc_diff_1_model_2_datasets_patient_bootstrap_ci_above_0
)
from hypothesis_testing_utils import (
    run_auc_diff_2_models_1_dataset_patient_permutation_test,
    run_auc_diff_1_model_2_datasets_patient_permutation_test
)

def check_for_nonstationarity(config_dict,
                              models,
                              valid_Xs,
                              valid_Ys,
                              valid_person_id_to_sample_idxs,
                              test_Xs,
                              test_Ys,
                              test_person_id_to_sample_idxs,
                              logger,
                              overwrite = False):
    '''
    Use validation set to check if set-up might be a candidate for multiple hypothesis
    Criteria:
    - AUC of current model evaluated on current year's data is at least .5
    - AUC of previous model evaluated on previous year's data is at least .5
    - At least 25 patients with outcome in current year's data and previous year's data
    - 90% bootstrap confidence interval of difference between current and previous year's AUC is above 0
    If criteria are satisfied, compute p-value using permutation test on test set.
    If there are fewer than 20 patients with outcome in test set, p-value is set to 1.
    Save computed p-values to csv
    @param config_dict: dict mapping str to int, str, etc containing set-up parameters and file paths
    @param models: list of sklearn models, one per year, None if insufficient samples to learn
    @param valid_Xs: list of csr matrices, sample features at each time point in validation set
    @param valid_Ys: list of np arrays, sample outcomes at each time point in validation set
    @param valid_person_id_to_sample_idxs: list of defaultdict mapping int to list of ints, person ID to sample indices 
                                           at each time point in validation set
    @param test_Xs: list of csr matrices, sample features at each time point in test set
    @param test_Ys: list of np arrays, sample outcomes at each time point in test set
    @param test_person_id_to_sample_idxs: list of defaultdict mapping int to list of ints, person ID to sample indices 
                                          at each time point in test set
    @param logger: logger, for INFO messages
    @param overwrite: bool, whether to overwrite summary files if they already exist
    @return: None
    '''
    start_time             = time.time()
    if config_dict['baseline']:
        nonstationary_filename = config_dict['output_file_header'] \
                               + 'baseline_nonstationarity_check.csv'
    else:
        nonstationary_filename = config_dict['output_file_header'] \
                               + 'nonstationarity_check.csv'
    if (not overwrite) and os.path.exists(nonstationary_filename):
        return
    
    num_years     = config_dict['num_years']
    starting_year = config_dict['starting_year'] + config_dict['starting_year_idx']
    assert len(models)   == num_years
    assert len(valid_Xs) == num_years
    assert len(valid_Ys) == num_years
    assert len(test_Xs)  == num_years
    assert len(test_Ys)  == num_years
    assert len(valid_person_id_to_sample_idxs) == num_years
    assert len(test_person_id_to_sample_idxs)  == num_years
    assert np.all(np.array([valid_Xs[year_idx].shape[0] == len(valid_Ys[year_idx])
                            for year_idx in range(num_years)]))
    assert np.all(np.array([test_Xs[year_idx].shape[0]  == len(test_Ys[year_idx])
                            for year_idx in range(num_years)]))
    for year_idx in range(num_years):
        if models[year_idx] is not None:
            assert valid_Xs[year_idx].shape[1] == models[year_idx].n_features_in_
            assert test_Xs[year_idx].shape[1]  == models[year_idx].n_features_in_
    
    # set up dataframe for holding p-values
    columns = ['Experiment', 'Outcome name', '2016', '2017', '2018', '2019', '2020']
    df_data = {'Experiment'  : [config_dict['experiment_name']],
               'Outcome name': [config_dict['outcome_name']]}
    for year in range(2016, starting_year + 1):
        df_data[str(year)] = [float('nan')]
    for year in range(starting_year + num_years, 2021):
        df_data[str(year)] = [float('nan')]
    
    this_start_time               = time.time()
    valid_min_outcome_req         = 25
    test_min_outcome_req          = 20
    satisfy_valid_min_outcome_req = []
    satisfy_test_min_outcome_req  = []
    # False if model does not exist that year
    satisfy_auc_min_req           = []
    for year_idx in range(num_years):
        # check if enough validation patients with outcome to evaluate year
        satisfy_valid_min_outcome_req.append(satisfy_patient_outcome_count_minimum(valid_Ys[year_idx], 
                                                                                   valid_person_id_to_sample_idxs[year_idx],
                                                                                   valid_min_outcome_req))
        
        # check if enough test patients with outcome to evaluate year
        satisfy_test_min_outcome_req.append(satisfy_patient_outcome_count_minimum(test_Ys[year_idx], 
                                                                                  test_person_id_to_sample_idxs[year_idx],
                                                                                  test_min_outcome_req))
        
        # check if model satisfies minimum AUC requirement
        if models[year_idx] is None:
            satisfy_auc_min_req.append(False)
        else:
            satisfy_auc_min_req.append(satisfy_auc_minimum(models[year_idx], 
                                                           valid_Xs[year_idx], 
                                                           valid_Ys[year_idx]))
    logger.info('Time to check minimum number of patients with outcome and minimum AUC requirements: ' 
                + str(time.time() - this_start_time) + ' seconds')
        
    for year_idx in range(1, num_years):
        year_str = str(starting_year + year_idx)
        
        # check if enough validation samples with each outcome to evaluate previous year
        if not satisfy_valid_min_outcome_req[year_idx - 1]:
            logger.info('Cannot run non-stationarity check for ' + year_str 
                        + ' since too few validation outcomes in previous year.')
            df_data[year_str] = [float('nan')]
            continue
        
        # check if enough validation samples with each outcome to evaluate current year
        if not satisfy_valid_min_outcome_req[year_idx]:
            logger.info('Cannot run non-stationarity check for ' + year_str 
                        + ' since too few validation outcomes in current year.')
            df_data[year_str] = [float('nan')]
            continue
        
        # check validation AUC criteria of previous model
        if not satisfy_auc_min_req[year_idx - 1]:
            logger.info('Cannot run non-stationarity check for ' + year_str 
                        + ' since model from previous year has AUC below .5.')
            df_data[year_str] = [float('nan')]
            continue
        
        # check validation AUC criteria of current model
        if (not config_dict['baseline']) and (not satisfy_auc_min_req[year_idx]):
            logger.info('Cannot run non-stationarity check for ' + year_str 
                        + ' since model from current year has AUC below .5.')
            df_data[year_str] = [float('nan')]
            continue
        
        # check validation AUC confidence interval overlap
        if config_dict['baseline']:
            if not satisfy_auc_diff_1_model_2_datasets_patient_bootstrap_ci_above_0(models[year_idx - 1],
                                                                                    valid_Xs[year_idx - 1],
                                                                                    valid_Ys[year_idx - 1],
                                                                                    valid_person_id_to_sample_idxs[year_idx - 1],
                                                                                    valid_Xs[year_idx],
                                                                                    valid_Ys[year_idx],
                                                                                    valid_person_id_to_sample_idxs[year_idx],
                                                                                    logger):
                logger.info('Will not test non-stationarity hypothesis for ' + year_str 
                            + ' since 90% bootstrap confidence interval for AUC difference of model on 2 validation sets'
                            + ' is not above 0.')
                df_data[year_str] = [float('nan')]
                continue
        else:
            if not satisfy_auc_diff_2_models_1_dataset_patient_bootstrap_ci_above_0(models[year_idx],
                                                                                    models[year_idx - 1],
                                                                                    valid_Xs[year_idx],
                                                                                    valid_Ys[year_idx],
                                                                                    valid_person_id_to_sample_idxs[year_idx],
                                                                                    logger):
                logger.info('Will not test non-stationarity hypothesis for ' + year_str 
                            + ' since 90% bootstrap confidence interval for AUC difference of 2 models on validation set'
                            + ' is not above 0.')
                df_data[year_str] = [float('nan')]
                continue
        
        # check if enough test samples to evaluate current year
        if not satisfy_test_min_outcome_req[year_idx]:
            logger.info('Non-stationarity hypothesis p-value set to 1 for ' + year_str 
                        + ' since too few test outcomes in current year.')
            df_data[year_str] = [1]
            continue
        
        # compute p-value
        #test_prev_year_curr_preds = models[year_idx - 1].predict_proba(test_Xs[year_idx])[:,1]
        if config_dict['baseline']:
            # check if enough test samples to evaluate previous year
            if not satisfy_test_min_outcome_req[year_idx - 1]:
                logger.info('Non-stationarity hypothesis p-value set to 1 for ' + year_str 
                            + ' since too few test outcomes in previous year.')
                df_data[year_str] = [1]
                continue
            
            #test_prev_year_prev_preds = models[year_idx - 1].predict_proba(test_Xs[year_idx - 1])[:,1]
            p_value = run_auc_diff_1_model_2_datasets_patient_permutation_test(models[year_idx - 1],
                                                                               test_Xs[year_idx - 1],
                                                                               test_Ys[year_idx - 1],
                                                                               test_person_id_to_sample_idxs[year_idx - 1],
                                                                               test_Xs[year_idx],
                                                                               test_Ys[year_idx],
                                                                               test_person_id_to_sample_idxs[year_idx],
                                                                               logger)
        else:
            #test_curr_year_curr_preds = models[year_idx].predict_proba(test_Xs[year_idx])[:,1]
            p_value = run_auc_diff_2_models_1_dataset_patient_permutation_test(models[year_idx],
                                                                               models[year_idx - 1],
                                                                               test_Xs[year_idx],
                                                                               test_Ys[year_idx],
                                                                               test_person_id_to_sample_idxs[year_idx],
                                                                               logger)
        logger.info('Non-stationarity hypothesis p-value for ' + year_str + ': ' + str(p_value))
        df_data[year_str] = [p_value]
        
    # save dataframe to csv
    df = pd.DataFrame(data    = df_data,
                      columns = columns)
    df.to_csv(nonstationary_filename, 
              index = False)
    logger.info('Saved p-values to ' + nonstationary_filename)
    logger.info('Checked for non-stationarity in ' + str(time.time() - start_time) + ' seconds')
       
if __name__ == '__main__':
    
    mp.set_start_method('spawn', force=True)
    
    start_time = time.time()
    parser     = create_parser()
    args       = parser.parse_args()
    assert args.outcome  in {'eol', 'condition', 'procedure', 'lab', 'lab_group'}
    feature_sets_readable = {'cond_proc': 'conditions + procedures',
                             'drugs'    : 'drugs',
                             'labs'     : 'labs',
                             'all'      : 'all features'}
    assert args.features in feature_sets_readable
    if args.outcome in {'condition', 'procedure', 'lab', 'lab_group'}:
        assert len(args.outcome_id) > 0
        assert ' ' not in args.outcome_id
        assert len(args.outcome_name) > 0
        if args.outcome in {'lab', 'lab_group'}:
            assert args.direction in {'low', 'high'}
            if args.outcome == 'lab_group':
                assert len(args.outcome_ids) > 0
    assert args.model in {'logreg', 'dectree', 'forest', 'xgboost'}
    if args.debug_size is not None:
        assert args.debug_size > 0
    assert args.fold in {0, 1, 2, 3}
    
    config_dict      = create_config_dict(args)
    logging_filename = config_dict['output_file_header'] + 'nonstationarity_check_' \
                     + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.log'
    logger           = set_up_logger('logger_main',
                                     logging_filename)
    multiprocessing_logging.install_mp_handler()
    logger.info('python3 run_nonstationarity_check.py'
                + ' --outcome='                 + args.outcome
                + ' --outcome_id='              + str(args.outcome_id)
                + ' --outcome_ids='             + str(args.outcome_ids)
                + ' --direction='               + args.direction
                + ' --features='                + args.features
                + ' --outcome_name='            + args.outcome_name
                + ' --model='                   + args.model
                + ' --debug_size='              + str(args.debug_size)
                + ' --baseline='                + str(args.baseline)
                + ' --omit_subpopulation='      + str(args.omit_subpopulation)
                + ' --single_year='             + str(args.single_year)
                + ' --feature_windows='         + str(args.feature_windows)
                + ' --fold='                    + str(args.fold))
    
    valid_metrics_filename             = config_dict['output_file_header'] + 'valid_metrics.json'
    valid_metrics_exists               = os.path.exists(valid_metrics_filename)
    test_metrics_filename              = config_dict['output_file_header'] + 'test_metrics.json'
    test_metrics_exists                = os.path.exists(test_metrics_filename)
    metric_plot_filename               = config_dict['output_file_header'] + '_auc_all_models.pdf'
    metric_plot_exists                 = os.path.exists(metric_plot_filename)
    if config_dict['baseline']:
        nonstationarity_check_filename = config_dict['output_file_header'] \
                                       + 'baseline_nonstationarity_check.csv'
    else:
        nonstationarity_check_filename = config_dict['output_file_header'] + 'nonstationarity_check.csv'
    nonstationarity_check_exists       = os.path.exists(nonstationarity_check_filename)
    if (not config_dict['baseline']) and (not args.omit_subpopulation):
        subpopulation_check_filename   = config_dict['output_dir'] + 'subpopulation_analysis_errors_dectree/' \
                                       + config_dict['experiment_name'] \
                                       + '_subpopulation_analysis_errors_dectree_nonstationarity_check.csv'
        # Force overwrite for now
        subpopulation_check_exists     = False #os.path.exists(subpopulation_check_filename)
    if test_metrics_exists:
        logger.info('Test metrics exist. Will not recreate.')
    if metric_plot_exists:
        logger.info('Metric plot exists. Will not recreate.')
    if nonstationarity_check_exists:
        logger.info('Non-stationarity check has been run before. Will not rerun.')
    if (not config_dict['baseline']) and (not args.omit_subpopulation) and subpopulation_check_exists:
        logger.info('Subpopulation checks have already been performed. Will not recreate.')
    if test_metrics_exists and metric_plot_exists and nonstationarity_check_exists \
    and (config_dict['baseline'] or args.omit_subpopulation or subpopulation_check_exists):
        sys.exit()
    
    '''
    To produce each file, the following items are needed:
    1. Test metrics: covariates, outcomes, models
    2. Plots: test metrics, can be loaded from file
    3. Nonstationarity check: test metrics, can be loaded from file
    4. Sub-population metrics: covariates, outcomes, possibly expanded/slightly different covariates, models
    '''
    if test_metrics_exists and (not metric_plot_exists):
        with open(test_metrics_filename, 'r') as f:
            json_contents = json.load(f)
        metrics         = json_contents['metrics_dict']
        metrics_to_plot = json_contents['metrics_to_plot']
        logger.info('Loaded metrics from ' + test_metrics_filename)
    
    new_models_learned   = False
    if (not test_metrics_exists) or (not nonstationarity_check_exists) \
    or ((not config_dict['baseline']) and (not args.omit_subpopulation) and (not subpopulation_check_exists)):
        # load data
        this_start_time  = time.time()
        Y_all_years_dict = load_outcomes(config_dict['dataset_file_header'],
                                         config_dict['num_years'],
                                         logger,
                                         starting_year_idx = config_dict['starting_year_idx'],
                                         test_dataset_file_header = config_dict['test_data_file_header'])
        logger.info('Time to load outcomes: ' + str(time.time() - this_start_time) + ' seconds')
        
        this_start_time                        = time.time()
        X_csr_all_years_dict, feature_names, _ = load_covariates(config_dict['dataset_file_header'],
                                                                 args.features,
                                                                 config_dict['num_years'],
                                                                 logger,
                                                                 starting_year_idx = config_dict['starting_year_idx'],
                                                                 test_dataset_file_header = config_dict['test_data_file_header'])
        logger.info('Time to load covariates: ' + str(time.time() - this_start_time) + ' seconds')
        
        X_csr_all_years_dicts = [X_csr_all_years_dict]
        
        if (not config_dict['baseline']) and (not args.omit_subpopulation) and (not subpopulation_check_exists):
            # load additional data for sub-populations
            all_features_indices  = {'logreg': 0, 'dectree': 0}
            for subpopulation_model_class in ['logreg', 'dectree']:
                if args.features != 'all' or subpopulation_model_class == 'dectree':
                    this_start_time    = time.time()
                    scale_age_back     = False
                    if subpopulation_model_class == 'dectree':
                        scale_age_back = True # scale before passing into dectree for better interpretation
                    X_csr_all_years_dict_all_features, all_feature_names, age_scaled_back \
                        = load_covariates(config_dict['dataset_file_header'],
                                          'all',
                                          config_dict['num_years'],
                                          logger,
                                          scale_age_back    = scale_age_back,
                                          starting_year_idx = config_dict['starting_year_idx'])
                    if args.features != 'all' or age_scaled_back:
                        X_csr_all_years_dicts.append(X_csr_all_years_dict_all_features)
                        all_features_indices[subpopulation_model_class] = len(X_csr_all_years_dicts) - 1
                    logger.info('Time to load covariates for ' + subpopulation_model_class + ' sub-population: '
                                + str(time.time() - this_start_time) + ' seconds')
        
        
        this_start_time = time.time()
        person_id_to_sample_idxs_dict \
            = load_person_id_to_sample_indices_mappings(config_dict['dataset_file_header'],
                                                        config_dict['num_years'],
                                                        starting_year_idx        = config_dict['starting_year_idx'],
                                                        test_dataset_file_header = config_dict['test_data_file_header'])
        logger.info('Time to load person ID to sample indices mappings: ' + str(time.time() - this_start_time) + ' seconds')
        
        X_csr_all_years_dict = X_csr_all_years_dicts[0]
        
        # load/learn models
        this_start_time      = time.time()
        if config_dict['model'] == 'dectree':
            _, feature_names_short, _ = get_all_feature_names_without_interactions(feature_names)
        else:
            feature_names_short       = None
        models, thresholds, new_models_learned \
            = learn_models_over_years(X_csr_all_years_dict['train'], 
                                      Y_all_years_dict['train'], 
                                      X_csr_all_years_dict['valid'],
                                      Y_all_years_dict['valid'],
                                      config_dict['model'],
                                      feature_names,
                                      config_dict['output_file_header'],
                                      config_dict['outcome_name'] + ' from ' 
                                      + feature_sets_readable[config_dict['feature_set']],
                                      logger,
                                      year_idxs           = range(config_dict['starting_year_idx'],
                                                                  config_dict['starting_year_idx'] + config_dict['num_years']),
                                      feature_names_short = feature_names_short)
        logger.info('Time to load or learn models: ' + str(time.time() - this_start_time) + ' seconds')
    
    # Now that everything has been loaded, produce each file if needed
    if (not args.single_year) and (new_models_learned or (not test_metrics_exists)):
        this_start_time          = time.time()
        metrics, metrics_to_plot = evaluate_models_on_future_years(models,
                                                                   thresholds,
                                                                   X_csr_all_years_dict['test'],
                                                                   Y_all_years_dict['test'],
                                                                   config_dict['output_file_header'],
                                                                   logger,
                                                                   overwrite = new_models_learned)
        logger.info('Time to evaluate models: ' + str(time.time() - this_start_time) + ' seconds')
        
    if new_models_learned or (not nonstationarity_check_exists):
        this_start_time = time.time()
                                
        check_for_nonstationarity(config_dict,
                                  models,
                                  X_csr_all_years_dict['valid'],
                                  Y_all_years_dict['valid'],
                                  person_id_to_sample_idxs_dict['valid'],
                                  X_csr_all_years_dict['test'],
                                  Y_all_years_dict['test'],
                                  person_id_to_sample_idxs_dict['test'],
                                  logger,
                                  overwrite = new_models_learned)
        logger.info('Time to check for non-stationarity: ' + str(time.time() - this_start_time) + ' seconds')
        
    if (not args.single_year) and (not args.omit_subpopulation) and (not config_dict['baseline']) \
    and (new_models_learned or (not subpopulation_check_exists)):
        for subpopulation_model_class in ['dectree']:#['logreg', 'dectree']:
            subpopulation_start_time          = time.time()
            subpopulation_remaining_args      = {'region_model'          : subpopulation_model_class,
                                                 'interactions'          : False,
                                                 'region_identifier'     : 'errors',
                                                 'single_feature_regions': False}
            subpopulation_config_dict         = create_subpopulation_config_dict(args,
                                                                                 subpopulation_remaining_args)
            X_csr_all_years_dict_all          = X_csr_all_years_dicts[all_features_indices[subpopulation_model_class]]
            subpopulation_age_scaled_back     = False
            if subpopulation_model_class == 'dectree':
                subpopulation_age_scaled_back = age_scaled_back
            run_subpopulation_analysis(subpopulation_config_dict,
                                       X_csr_all_years_dict_all,
                                       X_csr_all_years_dict,
                                       Y_all_years_dict,
                                       all_feature_names,
                                       person_id_to_sample_idxs_dict,
                                       subpopulation_age_scaled_back,
                                       models,
                                       thresholds,
                                       logger,
                                       overwrite = new_models_learned)
            logger.info('Time to run ' + subpopulation_model_class + ' sub-population check: ' 
                        + str(time.time() - subpopulation_start_time) + ' seconds')
    
    logger.info('Total time: ' + str(time.time() - start_time) + ' seconds')