import os
import json
import numpy as np
import time
import pandas as pd
from datetime import datetime
from itertools import product
import sys
from os.path import dirname, abspath, join

from summarize_nonstationarity_check_results import create_parser

sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from logging_utils import set_up_logger
from hypothesis_testing_utils import run_benjamini_hochberg
from nonstationarity_scan_metric_dict_utils import convert_str_to_num_in_metrics_dict

def summarize_baseline_results(parent_dir,
                               summary_dir,
                               logger,
                               incomplete    = False,
                               excluded_list = [],
                               clinical_signif_threshold = 0.01):
    '''
    Runs multiple hypothesis testing for baseline results across all set-ups and sub-populations
    Creates a csv containing sorted p-values, critical values, and comparison with non-baseline results
    @param parent_dir: str, path to directory containing nonstationarity scan experiment results
    @param summary_dir: str, path to directory for putting results and copying plots
    @param logger: logger, for INFO messages
    @param incomplete: bool, does not halt if hypothesis tests are missing for incomplete scans
    @param excluded_list: list of str, experiment directories to exclude from summary
    @param clinical_signif_threshold: float, threshold for AUC difference to count as clinically significant
    @return: None
    '''
    non_baseline_testing_csv = summary_dir + 'multiple_hypothesis_testing.csv'
    assert os.path.exists(non_baseline_testing_csv)
    
    start_time        = time.time()
    experiment_dirs   = os.listdir(parent_dir)
    df_columns        = ['Experiment', 'Outcome name', 'Region name', 'Year', 'P-value', 'AUC diff']
    df_data           = {col: [] for col in df_columns}
    subpopulation_dir = 'subpopulation_analysis_errors_dectree'
    
    all_files_present = True
    for experiment_dir in experiment_dirs:
        if not (experiment_dir.startswith('lab_') 
                or experiment_dir.startswith('condition_')
                or experiment_dir.startswith('procedure_')
                or experiment_dir.startswith('eol_')):
            continue
            
        if experiment_dir in excluded_list:
            continue
        
        check_file_ending        = '_baseline_nonstationarity_check.csv'
        nonstationarity_filename = parent_dir + experiment_dir + '/' + experiment_dir + check_file_ending
        
        test_metrics_file_ending = '_test_metrics.json'
        test_metrics_filename    = parent_dir + experiment_dir + '/' + experiment_dir + test_metrics_file_ending
        
        # log all missing files if cannot summarize
        if not os.path.exists(nonstationarity_filename):
            logger.info('Missing ' + nonstationarity_filename)
            all_files_present    = False
            continue
        
        if not os.path.exists(test_metrics_filename):
            logger.info('Missing ' + test_metrics_filename)
            all_files_present    = False
            continue
        
        if (not incomplete) and (not all_files_present):
            continue
        
        nonstationarity_df = pd.read_csv(nonstationarity_filename)
        for year in range(2016, 2021):
            if not pd.isnull(nonstationarity_df[str(year)].values[0]):
                with open(test_metrics_filename, 'r') as f:
                    test_metrics = convert_str_to_num_in_metrics_dict(json.load(f)['metrics_dict'])
                if experiment_dir.startswith('condition_'):
                    starting_year = 2017
                else:
                    starting_year = 2015
                
                year_idx = year - starting_year
                curr_metric = test_metrics[year_idx - 1][year_idx]['auc']
                prev_metric = test_metrics[year_idx - 1][year_idx - 1]['auc']
                df_data['Experiment'].append(nonstationarity_df['Experiment'].values[0])
                df_data['Outcome name'].append(nonstationarity_df['Outcome name'].values[0])
                df_data['Region name'].append(None)
                df_data['Year'].append(year)
                df_data['P-value'].append(nonstationarity_df[str(year)].values[0])
                df_data['AUC diff'].append(prev_metric - curr_metric)
    
    if (not incomplete) and (not all_files_present):
        logger.info('Cannot summarize results because some files are missing.')
        sys.exit()
    
    logger.info('Loaded all p-values in ' + str(time.time() - start_time) + ' seconds')
    
    start_time   = time.time()
    baseline_df  = pd.DataFrame(data    = df_data,
                                columns = df_columns)
    baseline_df  = run_benjamini_hochberg(baseline_df)
    baseline_csv = summary_dir + 'baseline_multiple_hypothesis_testing.csv'
    baseline_df.to_csv(baseline_csv,
                       index = False)
    logger.info('Saved multiple hypothesis testing results to ' + baseline_csv
                + ' in ' + str(time.time() - start_time) + ' seconds')
    if clinical_signif_threshold > 0:
        baseline_df = baseline_df.loc[baseline_df['AUC diff'] >= clinical_signif_threshold]
    
    # get statistics for each outcome type
    outcome_types      = ['condition', 'lab', 'procedure']
    years              = [year for year in range(2016, 2021)]
    year_counts        = {year: [] for year in years}
    for outcome_type, year in product(outcome_types, years):
        num_outcome_year = len(baseline_df.loc[np.logical_and(baseline_df['Year'] == year,
                                                              baseline_df['Experiment'].str.startswith(outcome_type))])
        year_counts[year].append(num_outcome_year)
    col_names                     = ['Outcome'] + years
    year_counts['Outcome']        = outcome_types
    outcome_year_counts_df        = pd.DataFrame(data    = year_counts,
                                                 columns = col_names)
    if clinical_signif_threshold > 0:
        baseline_counts_csv       = summary_dir + 'baseline_multiple_hypothesis_testing' \
                                  + '_clinical_signif_counts.csv'
    else:
        baseline_counts_csv       = summary_dir + 'baseline_multiple_hypothesis_testing_counts.csv'
    outcome_year_counts_df.to_csv(baseline_counts_csv,
                                  index = False)
    logger.info('Saved counts of accepted hypotheses to ' + baseline_counts_csv)
    logger.info('Aggregated statistics on accepted hypotheses in ' + str(time.time() - start_time) + ' seconds')
    
    # merge with non-baseline results to get statistics on differences
    start_time      = time.time()
    non_baseline_df = pd.read_csv(non_baseline_testing_csv)
    non_baseline_df = non_baseline_df.loc[pd.isnull(non_baseline_df['Region name'])]
    if clinical_signif_threshold > 0:
        non_baseline_df = non_baseline_df.loc[non_baseline_df['AUC diff'] >= clinical_signif_threshold]
    baseline_df     = baseline_df.merge(non_baseline_df,
                                        how      = 'outer',
                                        on       = ['Experiment', 'Outcome name', 'Region name', 'Year'],
                                        suffixes = [' baseline', ' non-baseline'])
    
    baseline_df['Accept baseline'].fillna(value       = 0,
                                          inplace     = True)
    baseline_df['Accept non-baseline'].fillna(value   = 0,
                                              inplace = True)
    
    baseline_df.sort_values(by        = ['Accept baseline', 'Accept non-baseline', 'Rank baseline', 'Rank non-baseline'],
                            ascending = [False,             False,                 True,            True],
                            inplace   = True)
    if clinical_signif_threshold > 0:
        baseline_comparison_csv = summary_dir + 'baseline_comparison_multiple_hypothesis_testing' \
                                + '_clinical_signif.csv'
    else:
        baseline_comparison_csv = summary_dir \
                                + 'baseline_comparison_multiple_hypothesis_testing.csv'
    baseline_df.to_csv(baseline_comparison_csv,
                       index = False)
    logger.info('Saved multiple hypothesis testing comparison between baseline and non-baseline to ' 
                + baseline_comparison_csv
                + ' in ' + str(time.time() - start_time) + ' seconds')
    
    num_both_accepted        = len(baseline_df.loc[np.logical_and(baseline_df['Accept baseline']     == 1,
                                                                  baseline_df['Accept non-baseline'] == 1)])
    logger.info(str(num_both_accepted) + ' hypotheses accepted by both baseline and non-baseline workflows')
    
    num_baseline_only_accepted        = len(baseline_df.loc[np.logical_and(baseline_df['Accept baseline']     == 1,
                                                                           baseline_df['Accept non-baseline'] == 0)])
    logger.info(str(num_baseline_only_accepted) + ' hypotheses accepted only by baseline workflow')
    
    num_non_baseline_only_accepted        = len(baseline_df.loc[np.logical_and(baseline_df['Accept baseline']     == 0,
                                                                               baseline_df['Accept non-baseline'] == 1)])
    logger.info(str(num_non_baseline_only_accepted) + ' hypotheses accepted only by non-baseline workflow')
    
if __name__ == '__main__':
    
    start_time   = time.time()
    parser       = create_parser()
    args         = parser.parse_args()
    assert not args.plot_only
    if len(args.exclude) > 0:
        excluded_list = args.exclude.split(',')
    else:
        excluded_list = []
    
    summary_name = 'experiments_selected_with_multiple_hypothesis_testing'
    summary_dir  = config.experiment_dir + summary_name + '/'
    assert os.path.exists(summary_dir)
    
    log_filename = summary_dir + 'baseline_' + summary_name + '_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.log'
    logger       = set_up_logger('logger_main',
                                 log_filename)
    logger.info('python3 summarize_baseline_nonstationarity_check_results.py'
                + ' --incomplete=' + str(args.incomplete)
                + ' --exclude='    + args.exclude)
    logger.info('Summarizing baseline experiments in ' + summary_dir)
    
    summarize_baseline_results(config.experiment_dir,
                               summary_dir,
                               logger,
                               incomplete    = args.incomplete,
                               excluded_list = excluded_list,
                               clinical_signif_threshold = 0.01)
    logger.info('Summarized baseline non-stationarity check results in ' + str(time.time() - start_time) + ' seconds')