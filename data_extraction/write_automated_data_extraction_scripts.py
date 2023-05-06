import argparse
import os
import sys
from os.path import dirname, abspath, join

sys.path.append(dirname(dirname(abspath(__file__))))
import config

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Write automated scripts for running data extraction.')
    parser.add_argument('--outcome', 
                        action='store', 
                        type=str, 
                        help='Specify outcome among condition or lab')
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
        
    output_dir         = 'data_extraction_scripts/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file_header = output_dir + 'run_nonstationarity_check_data_extraction_for_' + args.outcome + '_outcomes_batch'
        
    task_batch_size = 5
    batch_idx       = 0
    output_str  = ''
    for outcome_idx in range(len(outcomes_list)):
        outcome_line       = outcomes_list[outcome_idx]
        outcome_fields     = outcome_line.strip().split(' - ')
        outcome_output_str = 'python3 extract_data.py --outcome=' + args.outcome + ' --outcome_id=' + outcome_fields[0]\
                           + ' --outcome_name_readable=\"' + outcome_fields[1] + '\"'
        if args.outcome == 'lab':
            outcome_output_str += ' --direction=' + outcome_fields[2]
        outcome_output_str     += ' --omit_features --finalize\n'
        output_str             += outcome_output_str
        
        if outcome_idx % task_batch_size == task_batch_size - 1 or outcome_idx == len(outcomes_list) - 1:
            output_file  = output_file_header + str(batch_idx) + '.sh'
            with open(output_file, 'w') as f:
                f.write(output_str)
            batch_idx += 1
            output_str = ''