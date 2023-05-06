import argparse
import os

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Write automated scripts for running non-stationarity checks for procedures.')
    parser.add_argument('--baseline',
                        action  = 'store_true',
                        default = False,
                        help    = 'Specify whether to include baseline parameter in scripts')
    args = parser.parse_args()
        
    procedure_groups_dir = '../data_extraction/procedure_groups/'
    procedure_groups     = os.listdir(procedure_groups_dir)
    output_dir           = 'run_experiment_scripts/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if args.baseline:
        output_fileheader = output_dir + 'run_baseline_nonstationarity_check_for_procedure_outcomes_batch'
    else:
        output_fileheader = output_dir + 'run_nonstationarity_check_for_procedure_outcomes_batch'
    
    idx_in_batch = 0
    batch_idx    = 0
    batch_str    = ''
    for procedure_group in procedure_groups:
        group_filename = procedure_groups_dir + procedure_group
        if os.path.isdir(group_filename):
            # only looking for text files containing procedure groups
            continue
        with open(group_filename, 'r') as f:
            for i in range(5):
                # assumes lines 2-5 are the commands for running experiments with each feature set
                command = f.readline()
                if i < 4: #== 0: # assumes line 5 is feature set all
                    continue
                command = command[:-1] + ' --model=logreg\n'
                if args.baseline:
                    command = command[:-1] + ' --baseline\n'
                batch_str += command
        idx_in_batch += 1
        if idx_in_batch == 5:
            output_filename = output_fileheader + str(batch_idx) + '.sh'
            with open(output_filename, 'w') as f:
                f.write(batch_str)
            idx_in_batch = 0
            batch_idx   += 1
            batch_str    = ''
    if idx_in_batch != 0:
        output_filename = output_fileheader + str(batch_idx) + '.sh'
        with open(output_filename, 'w') as f:
            f.write(batch_str)