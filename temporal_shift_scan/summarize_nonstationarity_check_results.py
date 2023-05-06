import argparse
import os
import json
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import product

import sys
from os.path import dirname, abspath, join
sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from logging_utils import set_up_logger
from hypothesis_testing_utils import run_benjamini_hochberg
from nonstationarity_scan_metric_dict_utils import convert_str_to_num_in_metrics_dict

def count_accepted_hypotheses(accepted_df,
                              counts_csv):
    '''
    Count the number of accepted hypotheses in each outcome type category
    @param accepted_df: pandas DataFrame, contains Year, Experiment, Outcome columns
    @param counts_csv: str, path to csv file to save counts
    @return: None
    '''
    outcome_types      = ['condition', 'lab', 'procedure']
    years              = [year for year in range(2016, 2021)]
    year_counts        = {year: [] for year in years}
    for outcome_type, year in product(outcome_types, years):
        num_outcome_year = len(accepted_df.loc[np.logical_and(
            accepted_df['Year'] == year,
            accepted_df['Experiment'].str.startswith(outcome_type))])
        year_counts[year].append(num_outcome_year)
    col_names                     = ['Outcome'] + years
    year_counts['Outcome']        = outcome_types
    outcome_year_counts_df        = pd.DataFrame(data    = year_counts,
                                                 columns = col_names)
    outcome_year_counts_df.to_csv(counts_csv,
                                  index = False)

def summarize_results(parent_dir,
                      summary_dir,
                      logger,
                      incomplete    = False,
                      excluded_list = []):
    '''
    Runs multiple hypothesis testing across all set-ups and sub-populations
    Creates a csv containing sorted p-values and critical values
    For selected entire population hypotheses, copies plots of test metrics and cohort sizes
    For selected sub-population hypotheses, copies plots of test metrics in entire population, 
    inside region, and outside region; subpopulation cohort sizes; and subpopulation decision tree plot
    @param parent_dir: str, path to directory containing nonstationarity scan experiment results
    @param summary_dir: str, path to directory for putting results and copying plots
    @param logger: logger, for INFO messages
    @param incomplete: bool, does not halt if hypothesis tests are missing for incomplete scans, does not copy plots
    @param excluded_list: list of str, experiment directories to exclude from summary
    @return: None
    '''
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
        
        check_file_ending        = '_nonstationarity_check.csv'
        nonstationarity_filename = parent_dir + experiment_dir + '/' + experiment_dir + check_file_ending
        subpopulation_filename   = parent_dir + experiment_dir + '/' + subpopulation_dir + '/' + experiment_dir \
                                 + '_' + subpopulation_dir + check_file_ending
        
        test_metrics_file_ending = '_test_metrics.json'
        test_metrics_filename    = parent_dir + experiment_dir + '/' + experiment_dir + test_metrics_file_ending
        
        # log all missing files if cannot summarize
        experiment_missing_file     = False
        if not os.path.exists(nonstationarity_filename):
            logger.info('Missing ' + nonstationarity_filename)
            experiment_missing_file = True
            all_files_present   = False
        
        if not os.path.exists(subpopulation_filename):
            logger.info('Missing ' + subpopulation_filename)
            experiment_missing_file = True
            all_files_present   = False
            
        if not os.path.exists(test_metrics_filename):
            logger.info('Missing ' + test_metrics_filename)
            all_files_present    = False
            continue
        
        if (not incomplete) and (not all_files_present):
            continue
        
        if experiment_missing_file:
            continue
        
        if experiment_dir.startswith('condition_'):
            starting_year = 2017
        else:
            starting_year = 2015
        
        nonstationarity_df = pd.read_csv(nonstationarity_filename)
        subpopulation_df   = pd.read_csv(subpopulation_filename)
        for year in range(2016, 2021):
            if not pd.isnull(nonstationarity_df[str(year)].values[0]):
                with open(test_metrics_filename, 'r') as f:
                    test_metrics = convert_str_to_num_in_metrics_dict(json.load(f)['metrics_dict'])
                year_idx = year - starting_year
                curr_metric = test_metrics[year_idx][year_idx]['auc']
                prev_metric = test_metrics[year_idx - 1][year_idx]['auc']
                df_data['Experiment'].append(nonstationarity_df['Experiment'].values[0])
                df_data['Outcome name'].append(nonstationarity_df['Outcome name'].values[0])
                df_data['Region name'].append(None)
                df_data['Year'].append(year)
                df_data['P-value'].append(nonstationarity_df[str(year)].values[0])
                df_data['AUC diff'].append(curr_metric - prev_metric)
                
            for region_idx in range(len(subpopulation_df)):
                year_idx = year - starting_year
                year_region_name = 'in_year_' + str(year_idx) + '_dectree_region'
                if subpopulation_df['Region name'].values[region_idx] != year_region_name:
                    continue
                if not pd.isnull(subpopulation_df[str(year)].values[region_idx]):
                    subpopulation_test_metrics_filename = parent_dir + experiment_dir + '/' + subpopulation_dir + '/' \
                                                        + experiment_dir + '_' + subpopulation_dir + '_' + year_region_name \
                                                        + test_metrics_file_ending
                    with open(subpopulation_test_metrics_filename, 'r') as f:
                        subpopulation_test_metrics = convert_str_to_num_in_metrics_dict(json.load(f)['metrics_dict'])
                    curr_metric = subpopulation_test_metrics[year_idx][year_idx]['auc']
                    prev_metric = subpopulation_test_metrics[year_idx - 1][year_idx]['auc']
                    
                    df_data['Experiment'].append(subpopulation_df['Experiment'].values[region_idx])
                    df_data['Outcome name'].append(subpopulation_df['Outcome name'].values[region_idx])
                    df_data['Region name'].append(subpopulation_df['Region name'].values[region_idx])
                    df_data['Year'].append(year)
                    df_data['P-value'].append(subpopulation_df[str(year)].values[region_idx])
                    df_data['AUC diff'].append(curr_metric - prev_metric)
    
    if (not incomplete) and (not all_files_present):
        logger.info('Cannot summarize results because some files are missing.')
        sys.exit()
    
    logger.info('Loaded all p-values in ' + str(time.time() - start_time) + ' seconds')
    
    start_time  = time.time()
    testing_df  = pd.DataFrame(data    = df_data,
                               columns = df_columns)
    testing_df  = run_benjamini_hochberg(testing_df)
    testing_csv = summary_dir + 'multiple_hypothesis_testing.csv'
    testing_df.to_csv(testing_csv,
                      index = False)
    logger.info('Saved multiple hypothesis testing results to ' + testing_csv
                + ' in ' + str(time.time() - start_time) + ' seconds')
    
    # get statistics on accepted hypotheses
    start_time           = time.time()
    accepted_df          = testing_df.loc[testing_df['Accept']==1]
    num_accepted         = len(accepted_df)
    accepted_df_entire   = accepted_df.loc[pd.isnull(accepted_df['Region name'])]
    accepted_df_subpop   = accepted_df.loc[~pd.isnull(accepted_df['Region name'])]
    num_regions_accepted = len(accepted_df_subpop)
    logger.info(str(num_regions_accepted) + ' of ' + str(num_accepted) + ' accepted hypotheses are for sub-populations')
    
    # get statistics for each outcome type
    counts_csv        = summary_dir + 'multiple_hypothesis_testing_counts.csv'
    subpop_counts_csv = summary_dir \
                      + 'multiple_hypothesis_testing_subpopulation_counts.csv'
    count_accepted_hypotheses(accepted_df_entire,
                              counts_csv)
    count_accepted_hypotheses(accepted_df_subpop,
                              subpop_counts_csv)
    logger.info('Saved counts of accepted hypotheses to ' + counts_csv)
    logger.info('Saved counts of accepted sub-population hypotheses to ' + subpop_counts_csv)
    logger.info('Aggregated statistics on accepted hypotheses in ' + str(time.time() - start_time) + ' seconds')
        
def plot_auc_diff_distribution(summary_dir,
                               logger,
                               clinical_signif_threshold = .01):
    '''
    Plot distribution of AUC differences for non-stationary set-ups
    @param summary_dir: str, path to directory for putting results and copying plots
    @param logger: logger, for INFO messages
    @param clinical_signif_threshold: float, threshold for AUC difference to include in plot
    @return: None
    '''
    testing_df           = pd.read_csv(summary_dir + 'multiple_hypothesis_testing.csv')
    accepted_df          = testing_df.loc[testing_df['Accept']==1]
    accepted_df_entire   = accepted_df.loc[pd.isnull(accepted_df['Region name'])]
    accepted_df_subpop   = accepted_df.loc[~pd.isnull(accepted_df['Region name'])]
    logger.info('Number of non-stationary outcomes: ' + str(len(accepted_df_entire)))
    logger.info('Number of non-stationary sub-populations: ' + str(len(accepted_df_subpop)))
    
    auc_diffs            = [0.01, 0.02, 0.05]
    if clinical_signif_threshold not in auc_diffs:
        auc_diffs.append(clinical_signif_threshold)
    for auc_diff in auc_diffs:
        num_outcomes_above_auc_diff = len(accepted_df_entire.loc[accepted_df_entire['AUC diff'] >= auc_diff])
        logger.info('Number of non-stationary outcomes with AUC difference above ' + str(auc_diff) + ': ' 
                    + str(num_outcomes_above_auc_diff))
        
        num_subpops_above_auc_diff  = len(accepted_df_subpop.loc[accepted_df_subpop['AUC diff'] >= auc_diff])
        logger.info('Number of non-stationary sub-populations with AUC difference above ' + str(auc_diff) + ': '
                    + str(num_subpops_above_auc_diff))
        
    accepted_df_entire = accepted_df_entire.loc[accepted_df_entire['AUC diff'] >= clinical_signif_threshold]
    accepted_df_subpop = accepted_df_subpop.loc[accepted_df_subpop['AUC diff'] >= clinical_signif_threshold]
    
    # get statistics for each outcome type
    counts_csv        = summary_dir + 'multiple_hypothesis_testing_clinical_signif_counts.csv'
    subpop_counts_csv = summary_dir \
                      + 'multiple_hypothesis_testing_subpopulation_clinical_signif_counts.csv'
    count_accepted_hypotheses(accepted_df_entire,
                              counts_csv)
    count_accepted_hypotheses(accepted_df_subpop,
                              subpop_counts_csv)
    logger.info('Saved counts of accepted clinically significant hypotheses to ' + counts_csv)
    logger.info('Saved counts of accepted clinically significant sub-population hypotheses to ' + subpop_counts_csv)
    logger.info('Aggregated statistics on accepted hypotheses in ' + str(time.time() - start_time) + ' seconds')
    
    plot_filename = summary_dir + 'nonstationary_auc_diff_dist.pdf'
    
    plt.clf()
    plt.rc('font', 
           family = 'serif', 
           size   = 14)
    plt.rc('xtick', 
           labelsize = 12)
    plt.rc('ytick', 
           labelsize = 12)
    fig, ax = plt.subplots(nrows = 2,
                           ncols = 1,
                           sharex = True,
                           figsize = (6.4, 6.4))
    
    ax[0].hist(accepted_df_entire['AUC diff'].values,
               bins  = 29,
               range = [0.01, 0.3])
    ax[0].set_xlabel('AUC difference')
    ax[0].set_ylabel('Frequency')
    ax[0].set_title('Non-stationary outcomes')
    ax[0].set_xlim(left = 0, right = .3)
    
    ax[1].hist(accepted_df_subpop['AUC diff'].values,
               bins  = 29,
               range = [0.01, 0.3])
    ax[1].set_xlabel('AUC difference')
    ax[1].set_ylabel('Frequency')
    ax[1].set_title('Non-stationary sub-populations')
    ax[1].set_xlim(left = 0, right = .3)
    plt.tight_layout()
    plt.savefig(plot_filename)
    
def create_parser():
    '''
    Create an argument parser
    @return: argparse ArgumentParser
    '''
    parser = argparse.ArgumentParser(description='Summarize experiment results with multiple hypothesis testing.')
    parser.add_argument('--incomplete',
                        action  = 'store_true',
                        default = False,
                        help    = 'Specify whether missing hypothesis tests are okay.')
    parser.add_argument('--exclude',
                        action  = 'store',
                        type    = str,
                        default = '',
                        help    = 'Comma-separated list of experiment directories to exclude.')
    parser.add_argument('--plot_only',
                        action  = 'store_true',
                        default = False,
                        help    = 'Results have already been summarized. Only recreate plot.')
    return parser
    
if __name__ == '__main__':
    
    start_time   = time.time()
    parser       = create_parser()
    args         = parser.parse_args()
    if len(args.exclude) > 0:
        excluded_list = args.exclude.split(',')
    else:
        excluded_list = []
    
    summary_name = 'experiments_selected_with_multiple_hypothesis_testing'
    summary_dir  = config.experiment_dir + summary_name + '/'
    if (not args.plot_only) and os.path.exists(summary_dir):
        sys.exit(summary_name + ' directory already exists. Please move the directory.')
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    
    log_filename = summary_dir + summary_name + '_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.log'
    logger       = set_up_logger('logger_main',
                                 log_filename)
    logger.info('python3 summarize_nonstationarity_check_results.py'
                + ' --incomplete=' + str(args.incomplete)
                + ' --exclude='    + args.exclude)
    logger.info('Summarizing experiments in ' + summary_dir)
    
    if not args.plot_only:
        summarize_results(config.experiment_dir,
                          summary_dir,
                          logger,
                          incomplete    = args.incomplete,
                          excluded_list = excluded_list)
    
    plot_auc_diff_distribution(summary_dir,
                               logger)
    logger.info('Summarized non-stationarity check results in ' + str(time.time() - start_time) + ' seconds')