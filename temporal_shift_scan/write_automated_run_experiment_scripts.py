import argparse
import os
import sys
from os.path import dirname, abspath, join

sys.path.append(dirname(dirname(abspath(__file__))))
import config

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Write automated scripts for running non-stationarity checks.')
    parser.add_argument('--outcome', 
                        action  = 'store', 
                        type    = str, 
                        help    = 'Specify outcome among condition or lab')
    parser.add_argument('--baseline',
                        action  = 'store_true',
                        default = False,
                        help    = 'Specify whether to include baseline parameter in scripts.')
    args = parser.parse_args()
    assert args.outcome in {'condition', 'lab'}
    
    data_dir = config.outcome_data_dir
    if args.outcome == 'lab':
        outcomes_list_file = data_dir + 'top_abnormal_labs_in_cohort_abbreviated.txt'
    else:
        outcomes_list_file = data_dir + 'top_conditions_in_cohort_abbreviated.txt'
        
    with open(outcomes_list_file, 'r') as f:
        outcomes_list = f.readlines()
    if outcomes_list[0].startswith('concept_id - '):
        outcomes_list.pop(0) # remove title line
    
    experiment_scripts_dir   = 'run_experiment_scripts/'
    if not os.path.exists(experiment_scripts_dir):
        os.makedirs(experiment_scripts_dir)
    if args.baseline:
        experiment_scripts_fileheader = experiment_scripts_dir + 'run_baseline_nonstationarity_check_for_' + args.outcome \
                                      + '_outcomes_batch'
    else:
        experiment_scripts_fileheader = experiment_scripts_dir + 'run_nonstationarity_check_for_' + args.outcome \
                                      + '_outcomes_batch'
    
    feature_sets    = ['all']#['cond_proc', 'drugs', 'labs', 'all']
    task_batch_size = 5
    batch_idx       = 0
    output_str      = ''
    for outcome_idx in range(len(outcomes_list)):
        outcome_line         = outcomes_list[outcome_idx]
        outcome_fields       = outcome_line.strip().split(' - ')
        for feature_set in feature_sets:
            outcome_str      = 'python3 run_nonstationarity_check.py --outcome=' + args.outcome + ' --outcome_id=' \
                             + outcome_fields[0] + ' --outcome_name=\"' + outcome_fields[1] + '\"'
            if args.outcome == 'lab':
                outcome_str += ' --direction=' + outcome_fields[2]
            outcome_str     += ' --features=' + feature_set
            outcome_str     += ' --model=logreg'
            if args.baseline:
                outcome_str += ' --baseline'
            output_str      += outcome_str + '\n'
            
        if outcome_idx % task_batch_size == task_batch_size - 1 or outcome_idx == len(outcomes_list) - 1:
            experiment_scripts_file = experiment_scripts_fileheader + str(batch_idx) + '.sh'
            with open(experiment_scripts_file, 'w') as f:
                f.write(output_str)
            batch_idx += 1
            output_str = ''