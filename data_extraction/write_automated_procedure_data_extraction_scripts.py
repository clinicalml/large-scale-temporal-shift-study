import os

if __name__ == '__main__':
    
    output_dir           = 'data_extraction_scripts/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_fileheader    = output_dir + 'run_nonstationarity_check_data_extraction_for_procedure_outcomes_batch'
    
    procedure_groups_dir = 'procedure_groups/'
    procedure_groups     = os.listdir(procedure_groups_dir)
    
    idx_in_batch = 0
    batch_idx    = 0
    batch_str    = ''
    for procedure_group in procedure_groups:
        group_filename = procedure_groups_dir + procedure_group
        if os.path.isdir(group_filename):
            # only looking for text files containing procedure groups
            continue
        with open(group_filename, 'r') as f:
            # assumes line 0 contains the command for extracting data
            batch_str += f.readline()
        idx_in_batch  += 1
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