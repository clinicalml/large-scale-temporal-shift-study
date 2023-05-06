import argparse
import os
import sys
import pickle
import time
import matplotlib.pyplot as plt
from datetime import datetime
from os.path import dirname, abspath, join

from extract_omop_data import *

sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from h5py_utils import *
from logging_utils import set_up_logger

sys.path.append(join(dirname(dirname(abspath(__file__))), 'nonstationarity_scan'))
from load_data_for_nonstationarity_scan import load_outcomes

def create_parser():
    '''
    Create argument parser
    '''
    parser = argparse.ArgumentParser(description='Extract data for non-stationarity check.')
    parser.add_argument('--produce_cohort_only',
                        action  = 'store',
                        type    = str,
                        default = '',
                        help    = 'Produce only the cohort. Specify cohort for which eligibility window: 1_year or 3_years.')
    parser.add_argument('--outcome', 
                        action  = 'store', 
                        type    = str, 
                        default = '',
                        help    = 'Specify outcome among eol, condition, procedure, lab, lab_group')
    parser.add_argument('--outcome_id', 
                        action  = 'store', 
                        type    = str, 
                        default = '', 
                        help    = ('Specify ID of condition, procedure, or lab outcome. '
                                   'Comma-separated IDs for multiple procedures or labs.')
                       )
    parser.add_argument('--direction',
                        action  = 'store',
                        type    = str,
                        default = '',
                        help    = 'Specify low or high for abnormal lab outcome')
    parser.add_argument('--outcome_name',
                        action  = 'store',
                        type    = str,
                        default = '',
                        help    = 'Specify name of procedure or lab group outcome. No spaces allowed since used for file names.')
    parser.add_argument('--outcome_name_readable',
                        action  = 'store',
                        type    = str,
                        default = '',
                        help    = 'Specify name of condition, procedure, or lab outcome for plot title.')
    parser.add_argument('--omit_features',
                        action  = 'store_true',
                        default = False,
                        help    = 'Skip feature extraction code if only extract outcomes without finalize data')
    parser.add_argument('--finalize',
                        action  = 'store_true',
                        default = False,
                        help    = 'Run finalize step despite omit_features flag. Only use if features are already generated.')
    parser.add_argument('--cohort_plot_only',
                        action  = 'store_true',
                        default = False,
                        help    = 'Only create the cohort plot. Assumes data has been finalized. Will overwrite existing plot.')
    parser.add_argument('--person_id_mapping_only',
                        action  = 'store_true',
                        default = False,
                        help    = 'Only extract person to sample indices mappings.')
    parser.add_argument('--debug_size',
                        action  = 'store',
                        type    = int,
                        default = None,
                        help    = 'Specify number of people to extract data for debugging.')
    parser.add_argument('--feature_windows',
                        action  = 'store',
                        type    = str,
                        default = '30',
                        help    = 'Specify comma-separated list of feature window lengths to extract in days.')
    parser.add_argument('--fold',
                        action  = 'store',
                        type    = int,
                        default = 0,
                        help    = ('Specify which fold to extract features for. '
                                   'Folds 1, 2, and 3 must be run after 0 because the same feature set as 0 is used.'))
    return parser

if __name__ == '__main__':
    
    start_time = time.time()
    
    parser = create_parser()
    args   = parser.parse_args()
    if len(args.produce_cohort_only) > 0:
        assert args.produce_cohort_only in {'1_year', '3_years'}
        eligibility_time = args.produce_cohort_only
    else:
        assert args.outcome in {'eol', 'condition', 'procedure', 'lab', 'lab_group'}
        outcome_sql_params = dict()
        if args.outcome == 'eol':
            outcome_name = 'eol'
        elif args.outcome == 'condition':
            assert len(args.outcome_id)   > 0
            assert ',' not in args.outcome_id
            assert len(args.outcome_name_readable) > 0
            outcome_sql_params['outcome_id']   = args.outcome_id
            outcome_name = 'condition_' + args.outcome_id
        elif args.outcome == 'procedure':
            assert len(args.outcome_id)   > 0
            assert len(args.outcome_name) > 0
            assert len(args.outcome_name_readable) > 0
            outcome_sql_params['outcome_id']   = args.outcome_id
            assert ' ' not in args.outcome_name
            outcome_sql_params['outcome_name'] = args.outcome_name
            outcome_name = 'procedure_' + args.outcome_name
        elif args.outcome == 'lab':
            assert len(args.outcome_id)   > 0
            assert ',' not in args.outcome_id
            assert len(args.outcome_name_readable) > 0
            assert args.direction in {'low', 'high'}
            outcome_sql_params['outcome_id']   = args.outcome_id
            outcome_sql_params['direction']    = args.direction
            outcome_name = 'lab_' + args.outcome_id + '_' + args.direction
        else: # args.outcome == 'lab_group'
            assert len(args.outcome_id)   > 0
            assert len(args.outcome_name) > 0
            assert len(args.outcome_name_readable) > 0
            outcome_sql_params['outcome_id']   = args.outcome_id
            assert ' ' not in args.outcome_name
            outcome_sql_params['outcome_name'] = args.outcome_name
            outcome_sql_params['direction']    = args.direction
            outcome_name = 'lab_group_' + args.outcome_name + '_' + args.direction
        if args.debug_size is not None:
            outcome_sql_params['debug_suffix'] = '_debug' + str(args.debug_size)
        else:
            outcome_sql_params['debug_suffix'] = ''
    assert args.fold in {0, 1, 2, 3}
    if args.person_id_mapping_only:
        assert args.omit_features
    
    eol_version          = False
    eol_suffix           = ''
    if args.outcome == 'eol':
        eol_version      = True
        eol_suffix       = '_eol'
    debug_suffix         = ''
    if args.debug_size is not None:
        debug_suffix     = '_debug' + str(args.debug_size)
    feature_window_days  = [int(days) for days in args.feature_windows.split(',')]
    if len(feature_window_days) == 1 and feature_window_days[0] == 30:
        feature_window_suffix = ''
    else:
        feature_window_suffix = '_' + '_'.join(map(str, feature_window_days)) + 'day_features'
    
    logging_dir          = config.logging_dir
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    time_str             = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    if len(args.produce_cohort_only) == 0:
        logging_filename = logging_dir + outcome_name + debug_suffix + '_' + time_str + '_data_extraction.log'
    else:
        logging_filename = logging_dir + 'produce_cohort_' + eligibility_time + feature_window_suffix + eol_suffix \
                         + debug_suffix + '_' + time_str + '_data_extraction.log'
    logger               = set_up_logger('logger_main',
                                         logging_filename)
    logger.info('python3 extract_data.py'
                + ' --produce_cohort_only='    + args.produce_cohort_only
                + ' --outcome='                + args.outcome 
                + ' --outcome_id='             + args.outcome_id
                + ' --direction='              + args.direction
                + ' --outcome_name='           + args.outcome_name
                + ' --outcome_name_readable='  + args.outcome_name_readable
                + ' --omit_features='          + str(args.omit_features)
                + ' --finalize='               + str(args.finalize)
                + ' --cohort_plot_only='       + str(args.cohort_plot_only)
                + ' --person_id_mapping_only=' + str(args.person_id_mapping_only)
                + ' --debug_size='             + str(args.debug_size)
                + ' --feature_windows='        + str(args.feature_windows)
                + ' --fold='                   + str(args.fold))
    
    if len(args.produce_cohort_only) == 0:
        if args.outcome == 'condition':
            eligibility_time = '3_years'
            starting_year    = 2017
            num_years        = 4
            # account for smaller cohort size
            min_freq         = 100
        else:
            eligibility_time = '1_year'
            starting_year    = 2015
            num_years        = 6
            min_freq         = 300
        outcome_sql_params['window_name'] = eligibility_time
        
        if args.outcome == 'eol':
            outcome_sql_file          = ''
            outcome_specific_data_dir = config.outcome_data_dir + 'dataset_eol_outcomes' + feature_window_suffix \
                                      + debug_suffix + '/'
            outcomes_hf5_file         = outcome_specific_data_dir + 'outcomes_eol.hf5'
            outcome_table_name        = 'monthly_eligibility_' + eligibility_time + eol_suffix + debug_suffix
            cohort_plot_title         = 'Mortality'
        elif args.outcome == 'condition':
            outcome_sql_file          = 'select_condition_outcome.sql'
            outcome_specific_data_dir = config.outcome_data_dir + 'dataset_' + outcome_name + '_outcomes' \
                                      + feature_window_suffix + debug_suffix + '/'
            outcomes_hf5_file         = outcome_specific_data_dir + 'outcomes_' + outcome_name + '.hf5'
            outcome_table_name        = outcome_name + '_outcomes' + debug_suffix
            cohort_plot_title         = args.outcome_name_readable
        elif args.outcome == 'procedure':
            outcome_sql_file          = 'select_procedure_outcome.sql'
            outcome_specific_data_dir = config.outcome_data_dir + 'dataset_' + outcome_name + '_outcomes' \
                                      + feature_window_suffix + debug_suffix + '/'
            outcomes_hf5_file         = outcome_specific_data_dir + 'outcomes_' + outcome_name + '.hf5'
            outcome_table_name        = outcome_name + '_outcomes' + debug_suffix
            cohort_plot_dir           = args.outcome + '_' + args.outcome_id + '_outcomes_freq' + str(min_freq)
            cohort_plot_title         = args.outcome_name_readable
        elif args.outcome == 'lab':
            outcome_sql_file          = 'select_abnormal_lab_outcome.sql'
            outcome_specific_data_dir = config.outcome_data_dir + 'dataset_' + outcome_name + '_outcomes' \
                                      + feature_window_suffix + debug_suffix + '/'
            outcomes_hf5_file         = outcome_specific_data_dir + 'outcomes_' + outcome_name + '.hf5'
            outcome_table_name        = outcome_name + '_outcomes' + debug_suffix
            cohort_plot_title         = args.outcome_name_readable + ' ' + args.direction
        else: # args.outcome == 'lab_group'
            outcome_sql_file          = 'select_abnormal_lab_group_outcome.sql'
            outcome_specific_data_dir = config.outcome_data_dir + 'dataset_' + outcome_name + '_outcomes' \
                                      + feature_window_suffix + debug_suffix + '/'
            outcomes_hf5_file         = outcome_specific_data_dir + 'outcomes_' + outcome_name + '.hf5'
            outcome_table_name        = outcome_name + '_outcomes' + debug_suffix
            cohort_plot_title         = args.outcome_name_readable + ' ' + args.direction
        cohort_plot_file_header       = outcome_specific_data_dir + outcome_name + '_outcomes' + debug_suffix
        
    if args.cohort_plot_only:
        this_start_time     = time.time()
        dataset_file_header = outcome_specific_data_dir + 'fold' + str(args.fold) + '_freq' + str(min_freq)
        Y_all_years_dict    = load_outcomes(dataset_file_header,
                                            num_years)
        logger.info('Time to load outcomes: ' + str(time.time() - this_start_time) + ' seconds')
        this_start_time = time.time()
        if args.outcome == 'lab':
            lab_outcome_id = args.outcome_id
        else:
            lab_outcome_id = None
        fig, _ = plot_cohort_size_and_outcome_freq(cohort_plot_file_header + '_cohort_size_outcome_freq',
                                                   cohort_plot_title,
                                                   Y_all_years_dict,
                                                   starting_year,
                                                   logger,
                                                   lab_outcome_id   = lab_outcome_id,
                                                   overwrite        = True,
                                                   eligibility_time = eligibility_time,
                                                   debug_size       = args.debug_size)
        plt.close(fig)
        logger.info('Time to plot cohort sizes and outcome frequencies: ' + str(time.time() - this_start_time) + ' seconds')
        sys.exit()
        
    this_start_time = time.time()
    create_window_prediction_dates(eligibility_time.replace('_', ' '))
    logger.info('Time to create window-specific prediction dates table: ' + str(time.time() - this_start_time) + ' seconds')
    
    this_start_time       = time.time()
    eligibility_hf5_path  = config.omop_data_dir + 'eligibility_' + eligibility_time + eol_suffix + '.hf5'
    eligible_ids_hf5_path = config.omop_data_dir + 'eligible_ids_' + eligibility_time + eol_suffix  + '.hf5'
    monthly_eligibility, eligible_ids = extract_monthly_eligibility(eligibility_hf5_path, 
                                                                    eligible_ids_hf5_path,
                                                                    eligibility_time.replace('_', ' '),
                                                                    eol_version,
                                                                    logger)
    logger.info('Time to extract monthly eligibility: ' + str(time.time() - this_start_time) + ' seconds')
    
    this_start_time = time.time()
    cohort_pkl_path = config.omop_data_dir + 'cohort_' + eligibility_time + eol_suffix + debug_suffix + '.pkl'
    cohort = extract_omop_cohort(cohort_pkl_path,
                                 eligibility_time.replace('_', ' '),
                                 eol_version,
                                 logger,
                                 args.debug_size)
    logger.info('Time to extract cohort: ' + str(time.time() - this_start_time) + ' seconds')
    if len(args.produce_cohort_only) > 0:
        sys.exit()
    
    features_hf5_path            = config.omop_data_dir + 'features_' + eligibility_time + eol_suffix + debug_suffix + '.hf5'
    features_auxiliary_hf5_path  = config.omop_data_dir + 'features_auxiliary_' + eligibility_time + eol_suffix \
                                 + debug_suffix + '.hf5'
    dataset_json_dir      = 'nonstationarity_cohort_' + eligibility_time + eol_suffix + debug_suffix + '/'
    windowed_features_dir = config.omop_data_dir + 'windowed_features_' + eligibility_time + feature_window_suffix \
                          + eol_suffix + debug_suffix + '_separate_files/'
    if not os.path.exists(windowed_features_dir):
        os.makedirs(windowed_features_dir)
    windowed_features_fileheader = windowed_features_dir + 'windowed_features'
    
    if not args.omit_features:
        this_start_time = time.time()
        feature_matrix, times_list, person_ids = extract_features(cohort,
                                                                  features_hf5_path,
                                                                  features_auxiliary_hf5_path,
                                                                  dataset_json_dir,
                                                                  eligibility_time.replace('_', ' '),
                                                                  eol_version,
                                                                  logger,
                                                                  args.debug_size)
        logger.info('Time to extract features: ' + str(time.time() - this_start_time) + ' seconds')
        
        this_start_time = time.time()
        extract_windowed_features(dataset_json_dir,
                                  feature_matrix,
                                  times_list,
                                  feature_window_days,
                                  windowed_features_fileheader,
                                  eligibility_time.replace('_', ' '),
                                  logger)
        logger.info('Time to extract windowed features: ' + str(time.time() - this_start_time) + ' seconds')
    elif os.path.exists(features_auxiliary_hf5_path):
        this_start_time = time.time()
        auxiliary_data  = load_data_from_h5py(features_auxiliary_hf5_path)
        person_ids      = auxiliary_data['person_ids']
        logger.info('Time to load times list and person IDs: ' + str(time.time() - this_start_time) + ' seconds')
    else:
        this_start_time = time.time()
        _, _, person_ids = extract_features(cohort,
                                            features_hf5_path,
                                            features_auxiliary_hf5_path,
                                            dataset_json_dir,
                                            eligibility_time.replace('_', ' '),
                                            eol_version,
                                            logger,
                                            args.debug_size)
        logger.info('Time to extract features for times list and person IDs: ' 
                    + str(time.time() - this_start_time) + ' seconds')
    
    this_start_time = time.time()
    if not os.path.exists(outcome_specific_data_dir):
        os.makedirs(outcome_specific_data_dir)
    if not args.person_id_mapping_only:
        outcomes_all_prediction_dates = extract_outcomes(outcome_sql_file,
                                                         outcome_sql_params,
                                                         dataset_json_dir,
                                                         eligible_ids,
                                                         outcomes_hf5_file,
                                                         eligibility_time.replace('_', ' '),
                                                         logger)
        logger.info('Time to extract outcomes: ' + str(time.time() - this_start_time) + ' seconds')
    
    this_start_time = time.time()
    num_folds = 4
    if args.outcome == 'eol':
        data_split_hf5_header = outcome_specific_data_dir + 'folds' + str(num_folds) + '_idxs_eol'
    else:
        data_split_hf5_header = outcome_specific_data_dir + 'folds' + str(num_folds) + '_idxs_' + outcome_table_name 
    train_indicators, valid_indicators, test_indicators = split_data(person_ids, 
                                                                     outcome_table_name,
                                                                     4,
                                                                     True,
                                                                     data_split_hf5_header,
                                                                     eligibility_time.replace('_', ' '),
                                                                     logger)
    logger.info('Time to split data: ' + str(time.time() - this_start_time) + ' seconds')
    
    if args.outcome == 'condition':
        # change monthly_eligibility to exclude patients who have had condition before
        this_start_time = time.time()
        condition_monthly_eligibility_hf5_file = outcome_specific_data_dir + 'condition_' + args.outcome_id \
                                               + '_eligibility' + debug_suffix + '.hf5'
        reindexed_monthly_eligibility = extract_condition_monthly_eligibility(eligible_ids,
                                                                              outcome_table_name,
                                                                              condition_monthly_eligibility_hf5_file,
                                                                              dataset_json_dir,
                                                                              eligibility_time.replace('_', ' '),
                                                                              logger)
        logger.info('Time to extract condition monthly eligibility: ' + str(time.time() - this_start_time) + ' seconds')
    else:
        # reindex monthly_eligibility to only patients in cohort
        reindexed_eligibility_hf5_file = config.omop_data_dir + 'reindexed_eligibility_' + eligibility_time + eol_suffix \
                                       + debug_suffix + '.hf5'
        this_start_time = time.time()
        reindexed_monthly_eligibility = reindex_monthly_eligibility(monthly_eligibility,
                                                                    dataset_json_dir,
                                                                    reindexed_eligibility_hf5_file,
                                                                    logger)
        logger.info('Time to re-index monthly eligibility: ' + str(time.time() - this_start_time) + ' seconds')
    
    if args.fold == 0:
        dataset_output_file_header = outcome_specific_data_dir + 'fold' + str(args.fold) + '_freq' + str(min_freq)
    else:
        dataset_output_file_header = outcome_specific_data_dir + 'fold' + str(args.fold) \
                                   + '_using_fold0_freq' + str(min_freq)
    
    run_finalize_features = (not args.omit_features) or (args.omit_features and args.finalize)
    if args.person_id_mapping_only or run_finalize_features:
        get_person_id_to_sample_mapping(person_ids,
                                        reindexed_monthly_eligibility,
                                        train_indicators[args.fold],
                                        valid_indicators[args.fold],
                                        test_indicators,
                                        dataset_output_file_header,
                                        logger)
    
    if (not args.omit_features) or (args.omit_features and args.finalize):
        if args.fold == 0:
            Y_all_years_dict = finalize_data(windowed_features_fileheader,
                                             outcomes_all_prediction_dates, 
                                             reindexed_monthly_eligibility,
                                             train_indicators[args.fold], 
                                             valid_indicators[args.fold], 
                                             test_indicators,
                                             min_freq,
                                             dataset_output_file_header,
                                             logger)

            this_start_time = time.time()
            if args.outcome == 'lab':
                lab_outcome_id = args.outcome_id
            else:
                lab_outcome_id = None
            plot_cohort_size_and_outcome_freq(cohort_plot_file_header + '_cohort_size_outcome_freq',
                                              cohort_plot_title,
                                              Y_all_years_dict,
                                              starting_year,
                                              logger,
                                              lab_outcome_id   = lab_outcome_id,
                                              eligibility_time = eligibility_time,
                                              debug_size       = args.debug_size)
            logger.info('Time to plot cohort sizes and outcome frequencies: ' + str(time.time() - this_start_time) + ' seconds')
        else:
            fold0_file_header          = outcome_specific_data_dir + 'fold0_freq' + str(min_freq)
            gather_fold_data_with_other_fold_features(windowed_features_fileheader, 
                                                      outcomes_all_prediction_dates, 
                                                      reindexed_monthly_eligibility,
                                                      train_indicators[args.fold], 
                                                      valid_indicators[args.fold], 
                                                      fold0_file_header,
                                                      dataset_output_file_header,
                                                      logger)
        
    logger.info('Time to extract data: ' + str(time.time() - start_time) + ' seconds')