import os
import numpy as np
import pandas as pd
import time
import sys
import argparse
import json
from datetime import datetime
from scipy.stats import chi2_contingency
from os.path import dirname, abspath, join

sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from logging_utils import set_up_logger

def check_for_condition_dependent_models(num_top_coefs):
    '''
    Look for set-ups that satisfy these criteria:
    - Non-stationary in 2020
    - Have conditions in top coefficients of 2019 model
    - Hypothesis test for label shift does not have p-value below .05
    @param num_top_coefs: int, number of top positive and top negative coefficients to check for conditions
    @return: None
    '''
    assert num_top_coefs > 0
    logger.info('Looking for non-stationary set-ups in 2020 with conditions in top ' + str(num_top_coefs)
                + ' coefficients of 2019 model and label shift hypothesis test without significant p-value')
    
    testing_results_dir = config.experiment_dir + 'experiments_selected_with_multiple_hypothesis_testing/'
    output_file         = 'run_experiment_scripts/run_nonstationarity_check_drug_features_only_for_domain_shift.sh'
    output_str          = ''
    if os.path.exists(output_file):
        logger.info(output_file + ' already exists')
        return
    
    # load experiments with non-stationarity in 2020
    testing_results_filename = testing_results_dir + 'multiple_hypothesis_testing.csv'
    testing_results_df       = pd.read_csv(testing_results_filename)
    shift_2020_df            = testing_results_df.loc[np.logical_and(testing_results_df['Year'] == 2020,
                                                                     pd.isnull(testing_results_df['Region name']))]
    
    def extract_feature_type(feat_name):
        '''
        Extract feature type, e.g. condition
        If there are fewer than 3 ' - ' in name, original name is returned
        @param feat_name: str
        @return: str, extracted type
        '''
        feat_name_parts = feat_name.split(' - ')
        if len(feat_name_parts) < 4:
            return feat_name
        return feat_name_parts[1]

    for experiment_idx in range(len(shift_2020_df)):
        # load coefficients for that year
        experiment_name         = shift_2020_df['Experiment'].iloc[experiment_idx]
        outcome_name            = shift_2020_df['Outcome name'].iloc[experiment_idx]
        
        experiment_specific_dir = config.experiment_dir + experiment_name + '/'
        if experiment_name.startswith('condition_'):
            year_idx_for_2019   = 2
        else:
            year_idx_for_2019   = 4
        logreg_filename         = experiment_specific_dir + experiment_name + '_logistic_regression_time' \
                                + str(year_idx_for_2019) + '_coefficients.csv'
        coef_df                 = pd.read_csv(logreg_filename)
        
        # look for condition features among top coefficients
        coef_df.sort_values(by        = 'Coefficient',
                            ascending = False,
                            inplace   = True)
        top_coef_df                   = pd.concat((coef_df.head(num_top_coefs),
                                                   coef_df.tail(num_top_coefs)))
        top_coef_df['Feature type']   = top_coef_df['Feature'].apply(extract_feature_type)
        top_condition_coef_df         = top_coef_df.loc[np.logical_and(top_coef_df['Feature type'] == 'condition',
                                                                       top_coef_df['Coefficient'].abs() > 1e-2)]
        if len(top_condition_coef_df) == 0:
            logger.info('No conditions in top coefficients for ' + experiment_name + ' in 2019')
            continue
        
        # load outcome frequencies
        outcome_name_for_files          = experiment_name[:experiment_name.index('_from_')]
        experiment_data_dir             = config.outcome_data_dir + 'dataset_' + outcome_name_for_files + '/'
        outcome_frequency_json_filename = experiment_data_dir + outcome_name_for_files + '_cohort_size_outcome_freq.json'
        with open(outcome_frequency_json_filename, 'r') as f:
            outcome_frequency_stats     = json.load(f)
        outcome_counts                  = outcome_frequency_stats['Outcome count']
        cohort_sizes                    = outcome_frequency_stats['Cohort size']
        
        # test for label shift
        outcome_frequency_matrix        = np.empty((2, 2),
                                                   dtype = int)
        outcome_frequency_matrix[0,0]   = int(cohort_sizes[year_idx_for_2019] - outcome_counts[year_idx_for_2019])
        outcome_frequency_matrix[0,1]   = int(outcome_counts[year_idx_for_2019])
        outcome_frequency_matrix[1,0]   = int(cohort_sizes[year_idx_for_2019 + 1] - outcome_counts[year_idx_for_2019 + 1])
        outcome_frequency_matrix[1,1]   = int(outcome_counts[year_idx_for_2019 + 1])
        _, p_value, _, _                = chi2_contingency(outcome_frequency_matrix)
        if p_value <= .05:
            logger.info('Label shift in ' + experiment_name + ' in 2020')
            continue
        
        # add set-up to script
        experiment_name_parts = experiment_name.split('_')
        outcome_type          = experiment_name_parts[0]
        if outcome_type == 'procedure':
            outcome_id        = '_'.join(experiment_name_parts[1:experiment_name_parts.index('outcomes')])
        else:
            outcome_id        = experiment_name_parts[1]
        output_str           += 'python3 run_nonstationarity_check.py' \
                              + ' --outcome=' + outcome_type \
                              + ' --outcome_id=' + outcome_id
        if outcome_type == 'lab':
            direction         = experiment_name_parts[2]
            output_str       += ' --direction=' + direction
        output_str           += ' --outcome_name=\"' + outcome_name + '\"' \
                              + ' --features=drugs --model=logreg --omit_subpopulation --single_year=2020\n'
        
        logger.info('Added ' + experiment_name + ' to script for running experiments with only drug features')
        
    with open(output_file, 'w') as f:
        f.write(output_str)
    logger.info('Wrote script to ' + output_file)
    
def create_parser():
    '''
    Create an argument parser
    @return: argparse ArgumentParser
    '''
    parser = argparse.ArgumentParser(description = ('Find examples of domain shift.'))
    parser.add_argument('--num_top_coefs',
                        action  = 'store',
                        type    = int,
                        default = 20,
                        help    = 'Specify number of top positive coefficients to look for conditions in.')
    return parser

if __name__ == '__main__':
    
    parser       = create_parser()
    args         = parser.parse_args()
    
    logging_filename = config.logging_dir + 'find_domain_shift_examples_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") \
                     + '.log' 
    logger           = set_up_logger('logger_main',
                                     logging_filename)
    logger.info('python3 find_domain_shift_examples.py'
                + ' --num_top_coefs='    + str(args.num_top_coefs))
    
    check_for_condition_dependent_models(args.num_top_coefs)