import os
import sys
import numpy as np
import pandas as pd
import json
import time
import shutil
from collections import defaultdict

from plot_nonstationarity_check import plot_all_metrics
from learn_and_evaluate_models_over_years import evaluate_models_on_future_years

from os.path import dirname, abspath, join
sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
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

def evaluate_models_per_year_on_region(config_dict,
                                       X_csr_all_years,
                                       X_csr_all_years_specific,
                                       Y_all_years,
                                       data_split,
                                       models,
                                       thresholds,
                                       region_names_dict,
                                       get_region_indicators,
                                       logger,
                                       errors_all_years = None,
                                       overwrite        = False):
    '''
    Run non-stationarity check for samples inside and outside region
    i.e. evaluate original models only on samples inside and outside region in each of the following years
    @param config_dict: dict mapping str to int or str, settings such as file paths, plot titles, number of years
    @param X_csr_all_years: list of csr matrices, covariates for each year with all features
    @param X_csr_all_years_specific: list of csr matrices, covariates for each year 
                                     with only features in original models
    @param Y_all_years: list of np arrays, outcomes for each year
    @param data_split: str, name of data split, used for interactions cache file name
    @param models: list of sklearn models, models have predict_proba function, one model per year
    @param thresholds: list of floats, thresholds for predicted probabilities each year
    @param region_names_dict: dict mapping str to str, names of region and not in region for plot titles and file names
    @param get_region_indicators: function, takes in X and Y, outputs indicators or probabilities 
                                  for whether sample is in region,
                                  if probabilities, samples will be weighted by how likely they are in region
    @param logger: logger, for INFO messages
    @param errors_all_years: list of np arrays, indicators for which samples in test split are predicted incorrectly 
                             by the previous year's model but not by the current year's model, 
                             list over years starting at year 1,
                             statistics on proportion and number of errors in region only computed if provided,
                             could also be indicators for when previous year's model is not as well calibrated
    @param overwrite: bool, whether to overwrite output files if they already exist
    @return: 1. dict mapping int model year index
                to int evaluation year index
                to str metric name
                to float metric value,
                years without models or enough samples to evaluate are omitted,
                evaluated on samples in region
             2. same format as 1, evaluated on samples not in region
             3. list of floats, proportion of samples in region each year
    '''
    assert {'region_name_short', 'region_name_full', 'region_file_name',
            'not_region_name_short', 'not_region_name_full', 'not_region_file_name'}.issubset(set(region_names_dict.keys()))
    assert len(X_csr_all_years)          == config_dict['num_years']
    assert len(X_csr_all_years_specific) == config_dict['num_years']
    assert len(Y_all_years)              == config_dict['num_years']
    assert len(models)                   == config_dict['num_years']
    assert len(thresholds)               == config_dict['num_years']
    if errors_all_years is not None:
        assert len(errors_all_years)     == config_dict['num_years'] - 1
    for idx_year in range(config_dict['num_years']):
        num_samples = X_csr_all_years[idx_year].shape[0]
        assert X_csr_all_years_specific[idx_year].shape[0] == num_samples
        assert Y_all_years[idx_year].shape[0]              == num_samples
        if idx_year > 0 and errors_all_years is not None and errors_all_years[idx_year - 1] is not None:
            assert errors_all_years[idx_year - 1].shape[0] == num_samples
    
    # all_outputs_dir / experiment_name / subpopulation_header will be header of output metrics and plots
    region_header               = config_dict['output_file_header'] + region_names_dict['region_file_name'] + '_'
    region_metrics_filename     = region_header + 'test_metrics.json'
    not_region_header           = config_dict['output_file_header'] + region_names_dict['not_region_file_name'] + '_'
    not_region_metrics_filename = not_region_header + 'test_metrics.json'
    region_stats_filename       = config_dict['output_file_header'] + region_names_dict['region_file_name'] \
                                + '_region_stats.json'
    region_metrics_exist        = os.path.exists(region_metrics_filename)
    not_region_metrics_exist    = os.path.exists(not_region_metrics_filename)
    region_stats_exist          = os.path.exists(region_stats_filename)
    
    if overwrite or (not region_metrics_exist) or (not not_region_metrics_exist) or (not region_stats_exist):
        start_time                 = time.time()
        interaction_terms_dir      = config_dict['tmp_data_dir'] + config_dict['experiment_name'] + '/'
        if not os.path.exists(interaction_terms_dir):
            os.makedirs(interaction_terms_dir)
        interaction_terms_header   = interaction_terms_dir + 'interactions_'
        region_indicators          = [get_region_indicators(X_csr_all_years[idx_year],
                                                            Y_all_years[idx_year],
                                                            interaction_terms_header + data_split + '_year'
                                                            + str(config_dict['starting_year'] + idx_year) + '.hf5')
                                      for idx_year in range(config_dict['num_years'])]
        region_weighted            = False
        if not np.all(np.logical_or(region_indicators[0] == 0,
                                    region_indicators[0] == 1)):
            region_weighted        = True
            logger.info('Region is weighted')
        logger.info('Time to get region indicators: ' + str(time.time() - start_time) + ' seconds')
        
    if overwrite or (not region_metrics_exist) or (not region_stats_exist):
        if region_weighted:
            Y_all_years_in_region  = Y_all_years
            region_weights         = region_indicators
        else:
            region_indices         = [np.nonzero(region_indicators[idx_year])[0]
                                      for idx_year in range(config_dict['num_years'])]
            Y_all_years_in_region  = [Y_all_years[idx_year][region_indices[idx_year]]
                                      for idx_year in range(config_dict['num_years'])]
            region_weights         = None
    
    if (not overwrite) and region_metrics_exist:
        with open(region_metrics_filename, 'r') as f:
            json_contents      = json.load(f)
        region_metrics         = convert_str_to_num_in_metrics_dict(json_contents['metrics_dict'])
        region_metrics_to_plot = json_contents['metrics_to_plot']
        logger.info('Loaded sub-population ' + region_names_dict['region_name_full'] + ' metrics from ' 
                    + region_metrics_filename)
    else:
        # run non-stationarity check for samples in region
        this_start_time            = time.time()
        if region_weighted:
            X_all_years_in_region  = X_csr_all_years_specific
        else:
            X_all_years_in_region  = [X_csr_all_years_specific[idx_year][region_indices[idx_year]]
                                      for idx_year in range(config_dict['num_years'])]
        
        region_metrics, region_metrics_to_plot = evaluate_models_on_future_years(models,
                                                                                 thresholds,
                                                                                 X_all_years_in_region,
                                                                                 Y_all_years_in_region,
                                                                                 region_header,
                                                                                 logger,
                                                                                 overwrite = overwrite,
                                                                                 weights   = region_weights)
        logger.info('Time to evaluate sub-population ' + region_names_dict['region_name_full'] + ': '
                    + str(time.time() - this_start_time) + ' seconds')
        
    this_start_time   = time.time()
    region_plot_title = config_dict['plot_title'] + ' ' + region_names_dict['region_name_short']
    plot_all_metrics(region_metrics_to_plot,
                     region_plot_title,
                     region_header,
                     logger,
                     overwrite = overwrite)
    logger.info('Time to plot metrics for sub-population ' + region_names_dict['region_name_full'] + ': '
                + str(time.time() - this_start_time) + ' seconds')

    if (not overwrite) and not_region_metrics_exist:
        with open(not_region_metrics_filename, 'r') as f:
            json_contents          = json.load(f)
        not_region_metrics         = convert_str_to_num_in_metrics_dict(json_contents['metrics_dict'])
        not_region_metrics_to_plot = json_contents['metrics_to_plot']
        logger.info('Loaded sub-population ' + region_names_dict['not_region_name_full'] + ' metrics from ' 
                    + region_metrics_filename)
    else:
        # run non-stationarity check for samples outside region
        this_start_time             = time.time()
        if region_weighted:
            not_region_weights      = [1 - region_indicators[idx_year]
                                       for idx_year in range(config_dict['num_years'])]
            X_all_years_not_region  = X_csr_all_years_specific
            Y_all_years_not_region  = Y_all_years
        else:
            not_region_indices      = [np.nonzero(np.where(region_indicators[idx_year] == 0, 1, 0))[0]
                                       for idx_year in range(config_dict['num_years'])]
            X_all_years_not_region  = [X_csr_all_years_specific[idx_year][not_region_indices[idx_year]]
                                       for idx_year in range(config_dict['num_years'])]
            Y_all_years_not_region  = [Y_all_years[idx_year][not_region_indices[idx_year]]
                                       for idx_year in range(config_dict['num_years'])]
            not_region_weights      = None

        not_region_metrics, not_region_metrics_to_plot = evaluate_models_on_future_years(models,
                                                                                         thresholds,
                                                                                         X_all_years_not_region,
                                                                                         Y_all_years_not_region,
                                                                                         not_region_header,
                                                                                         logger, 
                                                                                         overwrite = overwrite,
                                                                                         weights   = not_region_weights)
        logger.info('Time to evaluate sub-population ' + region_names_dict['region_name_full'] + ': '
                    + str(time.time() - this_start_time) + ' seconds')
        
    this_start_time       = time.time()
    not_region_plot_title = config_dict['plot_title'] + ' ' + region_names_dict['not_region_name_short']
    plot_all_metrics(not_region_metrics_to_plot,
                     not_region_plot_title,
                     not_region_header,
                     logger,
                     overwrite = overwrite)
    logger.info('Time to plot metrics for sub-population ' + region_names_dict['not_region_name_full'] + ': '
                + str(time.time() - this_start_time) + ' seconds')
    
    if (not overwrite) and region_stats_exist:
        logger.info('Region stats were previously saved at ' + region_stats_filename)
        with open(region_stats_filename, 'r') as f:
            proportion_patients_in_region = json.load(f)['proportion_patients_in_region']
    else:
        # record number of samples, proportion of samples, number of errors, and proportion of errors inside region
        if region_weighted:
            region_num_patients           = [np.sum(region_weights[idx_year])
                                             for idx_year in range(config_dict['num_years'])]
        else:
            region_num_patients           = [len(Y_all_years_in_region[idx_year]) 
                                             for idx_year in range(config_dict['num_years'])]
        proportion_patients_in_region     = [region_num_patients[idx_year]/float(len(Y_all_years[idx_year]))
                                             for idx_year in range(config_dict['num_years'])]
        json_contents                     = {'region_num_patients'             : region_num_patients,
                                             'proportion_patients_in_region'   : proportion_patients_in_region}
        logger.info('# of patients in region each year: '         + ', '.join(map(str, region_num_patients)))
        logger.info('Proportion of samples in region each year: ' + ', '.join(map(str, proportion_patients_in_region)))
        
        if errors_all_years is not None:
            if region_weighted:
                region_error_counts       = [np.sum(np.multiply(errors_all_years[idx_year], region_weights[idx_year + 1]))
                                             if errors_all_years[idx_year] is not None
                                             else 0
                                             for idx_year in range(config_dict['num_years'] - 1)]
            else:
                region_error_counts       = [int(np.sum(errors_all_years[idx_year][region_indices[idx_year + 1]]))
                                             if errors_all_years[idx_year] is not None
                                             else 0
                                             for idx_year in range(config_dict['num_years'] - 1)]
            all_error_counts              = [np.sum(errors_all_years[idx_year])
                                             if errors_all_years[idx_year] is not None
                                             else 0
                                             for idx_year in range(config_dict['num_years'] - 1)]
            proportion_errors_in_region   = [region_error_counts[idx_year]/float(all_error_counts[idx_year])
                                             if all_error_counts[idx_year] > 0
                                             else 1
                                             for idx_year in range(config_dict['num_years'] - 1)]
            json_contents['region_error_counts']         = region_error_counts
            json_contents['proportion_errors_in_region'] = proportion_errors_in_region
            logger.info('# of errors in region each year: '           + ', '.join(map(str, region_error_counts)))
            logger.info('Proportion of errors in region each year: ' + ', '.join(map(str, proportion_errors_in_region)))
                                         
        with open(region_stats_filename, 'w') as f:
            json.dump(json_contents, f)
    
    return region_metrics, not_region_metrics, proportion_patients_in_region
    
def check_for_region_nonstationarity(config_dict,
                                     region_name,
                                     models,
                                     get_region_indicators,
                                     valid_region_Xs,
                                     valid_model_Xs,
                                     valid_Ys,
                                     valid_person_id_to_sample_idxs,
                                     test_region_Xs,
                                     test_model_Xs,
                                     test_Ys,
                                     test_person_id_to_sample_idxs,
                                     logger,
                                     overwrite = False,
                                     only_run_region_def_year = False):
    '''
    Use validation set to check if region might be a candidate for multiple hypothesis
    Criteria for enough samples:
    - Region contains between .1% and 75% of whole population in current year
    - At least 25 patients with outcome in current year's data and previous year's data inside and outside region
    - At least 25 patients without outcome in current year's data and previous year's data inside and outside region 
      (to ensure region is not too homogeneous)
    Criteria for current model does well enough overall and inside region:
    - AUC of current model evaluated on current year's data in entire population is at least .5
    - AUC of current model evaluated on current year's data in region is at least .5
    Criteria for previous model does well enough overall:
    - AUC of previous model evaluated on previous year's data in entire population is at least .5
    Criteria for AUC difference:
    - 90% bootstrap confidence interval of difference between current and previous year's AUC is above 0 in the region
    - 90% bootstrap confidence interval of difference between current and previous year's AUC is not above 0 outside the region
      OR current model's AUC outside the region is at most .5
    If criteria are satisfied, compute p-value using permutation test on test set in the region.
    If there are fewer than 20 patients with outcome or fewer than 20 patients without outcome inside region in test set, 
    p-value is set to 1.
    Save computed p-values to csv
    If baseline workflow is specified in config_dict, comparison is between performance of previous model on current data
    and previous data, and criteria pertaining to current model are dropped.
    Instead, check the AUC of the previous model evaluated on previous year's data in region is at least .5
    @param config_dict: dict mapping str to int, str, etc containing set-up parameters and file paths
    @param region_name: str, for file names
    @param models: list of sklearn models, one per year, None if insufficient samples to learn
    @param get_region_indicators: function that takes in X and Y and outputs binary indicators 
                                  for whether each sample is in the region
    @param valid_region_Xs: list of csr matrices, sample features at each time point in validation set,
                            used for getting region indicators
    @param valid_model_Xs: list of csr matrices, sample features at each time point in validation set,
                           used in original models
    @param valid_Ys: list of np arrays, sample outcomes at each time point in validation set
    @param valid_person_id_to_sample_idxs: list of defaultdict mapping int to list of ints, person ID to sample indices 
                                           at each time point in validation set
    @param test_region_Xs: list of csr matrices, sample features at each time point in test set,
                           used for getting region indicators
    @param test_model_Xs: list of csr matrices, sample features at each time point in test set,
                          used in original models
    @param test_person_id_to_sample_idxs: list of defaultdict mapping int to list of ints, person ID to sample indices 
                                          at each time point in test set
    @param test_Ys: list of np arrays, sample outcomes at each time point in test set
    @param logger: logger, for INFO messages
    @param overwrite: bool, whether to overwrite summary files if they already exist
    @param only_run_region_def_year: bool, whether to only test time point for which sub-population is defined
    @return: dataframe containing p-values
    '''
    start_time             = time.time()
    nonstationary_filename = config_dict['output_file_header'] + region_name \
                           + '_nonstationarity_check.csv'
    if (not overwrite) and os.path.exists(nonstationary_filename):
        df = pd.read_csv(nonstationary_filename)
        return df
    
    num_years = len(models)
    assert len(valid_region_Xs) == num_years
    assert len(valid_model_Xs)  == num_years
    assert len(valid_Ys)        == num_years
    assert len(test_region_Xs)  == num_years
    assert len(test_model_Xs)   == num_years
    assert len(test_Ys)         == num_years
    assert len(valid_person_id_to_sample_idxs) == num_years
    assert len(test_person_id_to_sample_idxs)  == num_years
    assert np.all(np.array([valid_region_Xs[year_idx].shape[0] == len(valid_Ys[year_idx])
                            for year_idx in range(num_years)]))
    assert np.all(np.array([test_region_Xs[year_idx].shape[0]  == len(test_Ys[year_idx])
                            for year_idx in range(num_years)]))
    assert np.all(np.array([valid_model_Xs[year_idx].shape[0]  == len(valid_Ys[year_idx])
                            for year_idx in range(num_years)]))
    assert np.all(np.array([test_model_Xs[year_idx].shape[0]   == len(test_Ys[year_idx])
                            for year_idx in range(num_years)]))
    for year_idx in range(num_years):
        if models[year_idx] is not None:
            assert valid_model_Xs[year_idx].shape[1] == models[year_idx].n_features_in_
            assert test_model_Xs[year_idx].shape[1]  == models[year_idx].n_features_in_
    
    # set up dataframe for holding p-values
    columns = ['Experiment', 'Outcome name', 'Region name', '2016', '2017', '2018', '2019', '2020']
    df_data = {'Experiment'  : [config_dict['experiment_name']],
               'Outcome name': [config_dict['outcome_name']],
               'Region name' : region_name}
    if num_years == 4:
        starting_year = 2017
        for year in ['2016', '2017']:
            df_data[str(year)] = [float('nan')]
    else:
        starting_year = 2015
    
    # get indicators for which samples are in region
    interaction_terms_dir      = config_dict['tmp_data_dir'] + config_dict['experiment_name'] + '/'
    if not os.path.exists(interaction_terms_dir):
        os.makedirs(interaction_terms_dir)
    if only_run_region_def_year:
        region_def_year_idx    = int(region_name[len('in_year_'):len('in_year_')+1])
        years_to_compute       = [region_def_year_idx - 1, region_def_year_idx]
        year_offset            = region_def_year_idx - 1
        for year in range(2016, 2021):
            if year - starting_year != region_def_year_idx:
                df_data[str(year)] = [float('nan')]
    else:
        years_to_compute       = range(num_years)
        year_offset            = 0
    interaction_terms_header   = interaction_terms_dir + 'interactions_'
    valid_region_indicators = [get_region_indicators(valid_region_Xs[year_idx],
                                                     valid_Ys[year_idx],
                                                     interaction_terms_header + 'valid_year'
                                                     + str(config_dict['starting_year'] + year_idx) + '.hf5')
                               for year_idx in years_to_compute]
    valid_region_idxs       = [np.nonzero(valid_region_indicators[year_idx])[0]
                               for year_idx in range(len(years_to_compute))]
    valid_not_region_idxs   = [np.nonzero(valid_region_indicators[year_idx] == 0)[0]
                               for year_idx in range(len(years_to_compute))]
    test_region_indicators  = [get_region_indicators(test_region_Xs[year_idx],
                                                     test_Ys[year_idx],
                                                     interaction_terms_header + 'test_year'
                                                     + str(config_dict['starting_year'] + year_idx) + '.hf5')
                               for year_idx in years_to_compute]
    test_region_idxs        = [np.nonzero(test_region_indicators[year_idx])[0]
                               for year_idx in range(len(years_to_compute))]
    
    # map person ID to indices among region samples and not region samples
    this_start_time                           = time.time()
    valid_person_id_to_region_sample_idxs     = []
    valid_person_id_to_not_region_sample_idxs = []
    test_person_id_to_region_sample_idxs      = []
    for year_idx in range(len(years_to_compute)):
        # map original sample index to region/not region sample index
        orig_valid_sample_idx_to_region_sample_idx     \
            = {valid_region_idxs[year_idx][region_sample_idx]: region_sample_idx
               for region_sample_idx in range(len(valid_region_idxs[year_idx]))}
        orig_valid_sample_idx_to_not_region_sample_idx \
            = {valid_not_region_idxs[year_idx][not_region_sample_idx]: not_region_sample_idx
               for not_region_sample_idx in range(len(valid_not_region_idxs[year_idx]))}
        orig_test_sample_idx_to_region_sample_idx     \
            = {test_region_idxs[year_idx][region_sample_idx]: region_sample_idx
               for region_sample_idx in range(len(test_region_idxs[year_idx]))}
        
        # construct map of person ID to indices among region/not region samples for this year
        # valid split
        year_valid_person_id_to_region_sample_idxs     = defaultdict(list)
        year_valid_person_id_to_not_region_sample_idxs = defaultdict(list)
        for person_id in valid_person_id_to_sample_idxs[year_offset + year_idx]:
            person_region_sample_indices \
                = np.array([orig_valid_sample_idx_to_region_sample_idx.get(orig_sample_idx, -1)
                            for orig_sample_idx in valid_person_id_to_sample_idxs[year_offset + year_idx][person_id]])
            person_region_sample_indices \
                = person_region_sample_indices[np.nonzero(person_region_sample_indices != -1)[0]]
            year_valid_person_id_to_region_sample_idxs[person_id] \
                = person_region_sample_indices.tolist()
            
            person_not_region_sample_indices \
                = np.array([orig_valid_sample_idx_to_not_region_sample_idx.get(orig_sample_idx, -1)
                            for orig_sample_idx in valid_person_id_to_sample_idxs[year_offset + year_idx][person_id]])
            person_not_region_sample_indices \
                = person_not_region_sample_indices[np.nonzero(person_not_region_sample_indices != -1)[0]]
            year_valid_person_id_to_not_region_sample_idxs[person_id] \
                = person_not_region_sample_indices.tolist()
        valid_person_id_to_region_sample_idxs.append(year_valid_person_id_to_region_sample_idxs)
        valid_person_id_to_not_region_sample_idxs.append(year_valid_person_id_to_not_region_sample_idxs)
        
        # test split
        year_test_person_id_to_region_sample_idxs = defaultdict(list)
        for person_id in test_person_id_to_sample_idxs[year_offset + year_idx]:
            person_region_sample_indices \
                = np.array([orig_test_sample_idx_to_region_sample_idx.get(orig_sample_idx, -1)
                            for orig_sample_idx in test_person_id_to_sample_idxs[year_offset + year_idx][person_id]])
            person_region_sample_indices \
                = person_region_sample_indices[np.nonzero(person_region_sample_indices != -1)[0]]
            year_test_person_id_to_region_sample_idxs[person_id] \
                = person_region_sample_indices.tolist()
        test_person_id_to_region_sample_idxs.append(year_test_person_id_to_region_sample_idxs)
        
    logger.info('Time to create person ID to sample mappings inside and outside region for validation set: '
                + str(time.time() - this_start_time) + ' seconds')
    
    this_start_time       = time.time()
    valid_min_outcome_req = 25
    test_min_outcome_req  = 20
    satisfy_valid_min_patient_with_outcome_region_req        = []
    satisfy_valid_min_patient_without_outcome_region_req     = []
    satisfy_valid_min_patient_with_outcome_not_region_req    = []
    satisfy_valid_min_patient_without_outcome_not_region_req = []
    satisfy_test_min_patient_with_outcome_region_req         = []
    satisfy_test_min_patient_without_outcome_region_req      = []
    # False if model does not exist that year
    satisfy_auc_min_req        = []
    satisfy_auc_min_region_req = []
    for year_idx in range(len(years_to_compute)):
        # check if enough validation patients with outcome in region to evaluate year
        satisfy_valid_min_patient_with_outcome_region_req.append(
            satisfy_patient_outcome_count_minimum(valid_Ys[year_offset + year_idx][valid_region_idxs[year_idx]], 
                                                  valid_person_id_to_region_sample_idxs[year_idx],
                                                  valid_min_outcome_req))
        
        # check if enough validation patients without outcome in region to evaluate year
        satisfy_valid_min_patient_without_outcome_region_req.append(
            satisfy_patient_outcome_count_minimum(np.where(valid_Ys[year_offset + year_idx][valid_region_idxs[year_idx]] 
                                                           == 1, 0, 1),
                                                  valid_person_id_to_region_sample_idxs[year_idx],
                                                  valid_min_outcome_req))
        
        # check if enough validation patients with outcome outside region to evaluate year
        satisfy_valid_min_patient_with_outcome_not_region_req.append(
            satisfy_patient_outcome_count_minimum(valid_Ys[year_offset + year_idx][valid_not_region_idxs[year_idx]], 
                                                  valid_person_id_to_not_region_sample_idxs[year_idx],
                                                  valid_min_outcome_req))
        
        # check if enough validation patients without outcome outside region to evaluate year
        satisfy_valid_min_patient_without_outcome_not_region_req.append(
            satisfy_patient_outcome_count_minimum(np.where(valid_Ys[year_offset + year_idx][valid_not_region_idxs[year_idx]] 
                                                           == 1, 0, 1),
                                                  valid_person_id_to_not_region_sample_idxs[year_idx],
                                                  valid_min_outcome_req))
        
        # check if enough test patients with outcome to evaluate year
        satisfy_test_min_patient_with_outcome_region_req.append(
            satisfy_patient_outcome_count_minimum(test_Ys[year_offset + year_idx][test_region_idxs[year_idx]], 
                                                  test_person_id_to_region_sample_idxs[year_idx],
                                                  test_min_outcome_req))
        
        # check if enough test patients without outcome in region to evaluate year
        satisfy_test_min_patient_without_outcome_region_req.append(
            satisfy_patient_outcome_count_minimum(np.where(test_Ys[year_offset + year_idx][test_region_idxs[year_idx]] 
                                                           == 1, 0, 1),
                                                  test_person_id_to_region_sample_idxs[year_idx],
                                                  test_min_outcome_req))
        
        # check if model satisfies minimum AUC requirement
        if models[year_offset + year_idx] is None:
            satisfy_auc_min_req.append(False)
            satisfy_auc_min_region_req.append(False)
        else:
            satisfy_auc_min_req.append(satisfy_auc_minimum(models[year_offset + year_idx], 
                                                           valid_model_Xs[year_offset + year_idx], 
                                                           valid_Ys[year_offset + year_idx]))
            satisfy_auc_min_region_req.append(satisfy_auc_minimum(models[year_offset + year_idx],
                                                                  valid_model_Xs[year_offset + year_idx][
                                                                      valid_region_idxs[year_idx]],
                                                                  valid_Ys[year_offset + year_idx][valid_region_idxs[year_idx]]))
    logger.info('Time to check minimum patients with outcome and AUC requirements for each year: '
                + str(time.time() - this_start_time) + ' seconds')
            
    for year_idx in range(1, len(years_to_compute)):
        year_str = str(starting_year + year_offset + year_idx)
        
        # check if region is between .1% and 75% of current year's data in validation set
        num_valid_samples = len(valid_Ys[year_offset + year_idx])
        valid_region_size = len(valid_region_idxs[year_idx])
        if valid_region_size < .001 * num_valid_samples or valid_region_size > .75 * num_valid_samples:
            logger.info('Cannot run non-stationarity check for ' + region_name + ' in ' + year_str 
                        + ' since region is too small in validation set of current year.')
            df_data[year_str] = [float('nan')]
            continue
        
        # check if enough validation patients with outcome inside region to evaluate previous year
        if not satisfy_valid_min_patient_with_outcome_region_req[year_idx - 1]:
            logger.info('Cannot run non-stationarity check for ' + region_name + ' in ' + year_str 
                        + ' since too few validation patients with outcome inside region in previous year.')
            df_data[year_str] = [float('nan')]
            continue
            
        # check if enough validation patients without outcome inside region to evaluate previous year
        if not satisfy_valid_min_patient_without_outcome_region_req[year_idx - 1]:
            logger.info('Cannot run non-stationarity check for ' + region_name + ' in ' + year_str 
                        + ' since too few validation patients without outcome inside region in previous year.')
            df_data[year_str] = [float('nan')]
            continue
        
        # check if enough validation patients with outcome outside region to evaluate previous year
        if not satisfy_valid_min_patient_with_outcome_not_region_req[year_idx - 1]:
            logger.info('Cannot run non-stationarity check for ' + region_name + ' in ' + year_str 
                        + ' since too few validation patients with outcome outside region in previous year.')
            df_data[year_str] = [float('nan')]
            continue
            
        # check if enough validation samples without outcome outside region to evaluate previous year
        if not satisfy_valid_min_patient_without_outcome_not_region_req[year_idx - 1]:
            logger.info('Cannot run non-stationarity check for ' + region_name + ' in ' + year_str 
                        + ' since too few validation samples without outcome outside region in previous year.')
            df_data[year_str] = [float('nan')]
            continue
            
        # check if enough validation patients with outcome inside region to evaluate current year
        if not satisfy_valid_min_patient_with_outcome_region_req[year_idx]:
            logger.info('Cannot run non-stationarity check for ' + region_name + ' in ' + year_str 
                        + ' since too few validation patients with outcome inside region in current year.')
            df_data[year_str] = [float('nan')]
            continue
            
        # check if enough validation patients without outcome inside region to evaluate current year
        if not satisfy_valid_min_patient_without_outcome_region_req[year_idx]:
            logger.info('Cannot run non-stationarity check for ' + region_name + ' in ' + year_str 
                        + ' since too few validation patients without outcome inside region in current year.')
            df_data[year_str] = [float('nan')]
            continue
        
        # check if enough validation patients with outcome outside region to evaluate current year
        if not satisfy_valid_min_patient_with_outcome_not_region_req[year_idx]:
            logger.info('Cannot run non-stationarity check for ' + region_name + ' in ' + year_str 
                        + ' since too few validation patients with outcome outside region in current year.')
            df_data[year_str] = [float('nan')]
            continue
        
        # check if enough validation samples without outcome outside region to evaluate current year
        if not satisfy_valid_min_patient_without_outcome_not_region_req[year_idx]:
            logger.info('Cannot run non-stationarity check for ' + region_name + ' in ' + year_str 
                        + ' since too few validation patients without outcome outside region in current year.')
            df_data[year_str] = [float('nan')]
            continue
        
        # check validation AUC criteria of previous model overall
        if not satisfy_auc_min_req[year_idx - 1]:
            logger.info('Cannot run non-stationarity check for ' + region_name + ' in ' + year_str 
                        + ' since validation AUC of model from previous year is below .5.')
            df_data[year_str] = [float('nan')]
            continue
        
        # Baseline: check validation AUC criteria of previous model in region
        if config_dict['baseline'] and (not satisfy_auc_min_region_req[year_idx - 1]):
            logger.info('Cannot run non-stationarity check for ' + region_name + ' in ' + year_str
                        + ' since validation AUC of model from previous year in region is below .5.')
            df_data[year_str] = [float('nan')]
            continue
        
        # Non-baseline: check validation AUC criteria of current model overall
        if (not config_dict['baseline']) and (not satisfy_auc_min_req[year_idx]):
            logger.info('Cannot run non-stationarity check for ' + region_name + ' in ' + year_str 
                        + ' since validation AUC of model from current year is below .5.')
            df_data[year_str] = [float('nan')]
            continue
        
        # Non-baseline: check validation AUC criteria of current model in region
        if (not config_dict['baseline']) and (not satisfy_auc_min_region_req[year_idx]):
            logger.info('Cannot run non-stationarity check for ' + region_name + ' in ' + year_str 
                        + ' since validation AUC of model from current year in region is below .5.')
            df_data[year_str] = [float('nan')]
            continue
        
        # check validation AUC confidence interval higher in region
        if config_dict['baseline']:
            if not satisfy_auc_diff_1_model_2_datasets_patient_bootstrap_ci_above_0(
                models[year_offset + year_idx - 1],
                valid_model_Xs[year_offset + year_idx - 1][valid_region_idxs[year_idx - 1]],
                valid_Ys[year_offset + year_idx - 1][valid_region_idxs[year_idx - 1]],
                valid_person_id_to_region_sample_idxs[year_idx - 1],
                valid_model_Xs[year_offset + year_idx][valid_region_idxs[year_idx]],
                valid_Ys[year_offset + year_idx][valid_region_idxs[year_idx]],
                valid_person_id_to_region_sample_idxs[year_idx],
                logger
            ):
                logger.info('Will not run non-stationarity check for ' + region_name + ' in ' + year_str 
                            + ' since 90% bootstrap confidence interval for AUC difference of model on 2 validation sets'
                            + ' inside region is not above 0.')
                df_data[year_str] = [float('nan')]
                continue
        else:
            if not satisfy_auc_diff_2_models_1_dataset_patient_bootstrap_ci_above_0(
                models[year_offset + year_idx],
                models[year_offset + year_idx - 1],
                valid_model_Xs[year_offset + year_idx][valid_region_idxs[year_idx]],
                valid_Ys[year_offset + year_idx][valid_region_idxs[year_idx]],
                valid_person_id_to_region_sample_idxs[year_idx],
                logger
            ):
                logger.info('Will not run non-stationarity check for ' + region_name + ' in ' + year_str 
                            + ' since 90% bootstrap confidence interval for AUC difference of 2 models on validation set'
                            + ' inside region is not above 0.')
                df_data[year_str] = [float('nan')]
                continue
        
        # check validation AUC confidence interval not higher outside region
        if config_dict['baseline']:
            if satisfy_auc_diff_1_model_2_datasets_patient_bootstrap_ci_above_0(
                models[year_offset + year_idx - 1],
                valid_model_Xs[year_offset + year_idx - 1][valid_not_region_idxs[year_idx - 1]],
                valid_Ys[year_offset + year_idx - 1][valid_not_region_idxs[year_idx - 1]],
                valid_person_id_to_not_region_sample_idxs[year_idx - 1],
                valid_model_Xs[year_offset + year_idx][valid_not_region_idxs[year_idx]],
                valid_Ys[year_offset + year_idx][valid_not_region_idxs[year_idx]],
                valid_person_id_to_not_region_sample_idxs[year_idx],
                logger
            ):
                logger.info('Will not run non-stationarity check for ' + region_name + ' in ' + year_str 
                            + ' since 90% bootstrap confidence interval for AUC difference of model on 2 validation sets'
                            + ' outside region is also above 0.')
                df_data[year_str] = [float('nan')]
                continue
        else:
            if satisfy_auc_diff_2_models_1_dataset_patient_bootstrap_ci_above_0(
                models[year_offset + year_idx],
                models[year_offset + year_idx - 1],
                valid_model_Xs[year_offset + year_idx][valid_not_region_idxs[year_idx]],
                valid_Ys[year_offset + year_idx][valid_not_region_idxs[year_idx]],
                valid_person_id_to_not_region_sample_idxs[year_idx],
                logger
            ):
                logger.info('Will not run non-stationarity check for ' + region_name + ' in ' + year_str 
                            + ' since 90% bootstrap confidence interval for AUC difference of 2 models on validation set'
                            + ' outside region is also above 0.')
                df_data[year_str] = [float('nan')]
                continue
        
        # check if enough test patients with outcome inside region to evaluate current year
        if not satisfy_test_min_patient_with_outcome_region_req[year_idx]:
            logger.info('Non-stationarity hypothesis p-value set to 1 for ' + region_name + ' in ' + year_str 
                        + ' since too few test patients with outcome inside region in current year.')
            df_data[year_str] = [1]
            continue
            
        # check if enough test samples without outcome inside region to evaluate current year
        if not satisfy_test_min_patient_without_outcome_region_req[year_idx]:
            logger.info('Non-stationarity hypothesis p-value set to 1 for ' + region_name + ' in ' + year_str 
                        + ' since too few test patients without outcome inside region in current year.')
            df_data[year_str] = [1]
            continue
        
        # compute p-value
        if config_dict['baseline']:
            # check if enough test patients with outcome inside region to evaluate previous year
            if not satisfy_test_min_patient_with_outcome_region_req[year_idx - 1]:
                logger.info('Non-stationarity hypothesis p-value set to 1 for ' + region_name + ' in ' + year_str 
                            + ' since too few test patients with outcome inside region in previous year.')
                df_data[year_str] = [1]
                continue
            
            # check if enough test patients without outcome inside region to evaluate previous year
            if not satisfy_test_min_patient_without_outcome_region_req[year_idx - 1]:
                logger.info('Non-stationarity hypothesis p-value set to 1 for ' + region_name + ' in ' + year_str 
                            + ' since too few test patients without outcome in region in previous year.')
                df_data[year_str] = [1]
                continue
            
            p_value = run_auc_diff_1_model_2_datasets_patient_permutation_test(
                models[year_offset + year_idx - 1],
                test_model_Xs[year_offset + year_idx - 1][test_region_idxs[year_idx - 1]],
                test_Ys[year_offset + year_idx - 1][test_region_idxs[year_idx - 1]],
                test_person_id_to_region_sample_idxs[year_idx - 1],
                test_model_Xs[year_offset + year_idx][test_region_idxs[year_idx]],
                test_Ys[year_offset + year_idx][test_region_idxs[year_idx]],
                test_person_id_to_region_sample_idxs[year_idx],
                logger)
        else:
            p_value = run_auc_diff_2_models_1_dataset_patient_permutation_test(
                models[year_offset + year_idx],
                models[year_offset + year_idx - 1],
                test_model_Xs[year_offset + year_idx][test_region_idxs[year_idx]],
                test_Ys[year_offset + year_idx][test_region_idxs[year_idx]],
                test_person_id_to_region_sample_idxs[year_idx],
                logger)
        
        logger.info('Non-stationarity hypothesis p-value for ' + region_name + ' in ' + year_str + ': ' + str(p_value))
        df_data[year_str] = [p_value]
        
    # save dataframe to csv
    df = pd.DataFrame(data    = df_data,
                      columns = columns)
    df.to_csv(nonstationary_filename, 
              index = False)
    logger.info('Saved p-values to ' + nonstationary_filename)
    logger.info('Checked for region non-stationarity in ' + str(time.time() - start_time) + ' seconds')
    return df