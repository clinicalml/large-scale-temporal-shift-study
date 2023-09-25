import argparse
import os
import sqlalchemy
import sys
from os.path import dirname, abspath, join

sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from db_utils import session_scope

def create_parser():
    '''
    Create argument parser for group name and inclusion-exclusion criteria
    @return: ArgumentParser
    '''
    parser = argparse.ArgumentParser(description='Extract procedure concepts for a group.')
    parser.add_argument('--include', 
                        action='store', 
                        type=str, 
                        help=('Specify %-separated strings to include in procedure concept name '
                              '(concept is included if any of these are in name).')
                       )
    parser.add_argument('--exclude',
                        action='store', 
                        type=str, 
                        help=('Specify %-separated strings to exclude from procedure concept name '
                              '(concept is excluded if any of these are in name)'),
                        default='')
    parser.add_argument('--group_name',
                        action='store',
                        type=str,
                        help='Specify name of procedure group for output file names. No spaces allowed')
    parser.add_argument('--group_name_readable',
                        action='store',
                        type=str,
                        help='Specify name of procedure group for plots')
    return parser

def extract_procedure_group(included_list,
                            excluded_list,
                            args):
    '''
    Extract group of procedures based on inclusion and exclusion criteria
    Write a file with script commands and procedure list
    @param included_list: list of str, procedure names with any of these are included
    @param excluded_list: list of str, procedure names with any of these are excluded
    @param args: arguments from ArgumentParser
    @return: None
    '''
    proc_sql = ('SELECT concept_id, '
                       'concept_name, '
                       'concept_count '
                'FROM {schema_name}.cohort_procedure_outcome_counts '
                'WHERE ('
               )
    for included_str in included_list:
        proc_sql += 'concept_name ILIKE \'%' + included_str.lower() + '%\' OR '
    proc_sql = proc_sql[:-4] + ') '
    for excluded_str in excluded_list:
        proc_sql += 'AND concept_name NOT ILIKE \'%' + excluded_str.lower() + '%\' '
    proc_sql = proc_sql[:-1] + ';'
    proc_sql = proc_sql.format(schema_name = config.nonstationarity_schema_name)
    print(proc_sql)
    
    engine = config.create_sqlalchemy_engine()
    with session_scope(engine) as session:
        proc_results = session.execute(sqlalchemy.text(proc_sql))
        session.commit()
    
    data_extraction_str = 'python3 extract_data.py --outcome=procedure --outcome_id='
    output_str = ''
    for result in proc_results:
        output_str          += str(result[0]) + ' - ' + str(result[1]) + ' - ' + str(result[2]) + '\n'
        data_extraction_str += str(result[0]) + ','
    data_extraction_str = data_extraction_str[:-1] + ' --outcome_name=' + args.group_name \
                        + ' --outcome_name_readable=\'' + args.group_name_readable + '\'' \
                        + ' --omit_features --finalize\n'
    
    feature_sets   = ['cond_proc', 'drugs', 'labs', 'all']
    experiment_str = ''
    for feature_set in feature_sets:
        this_experiment_str = 'python3 run_nonstationarity_check.py --outcome=procedure --outcome_id=' + args.group_name \
                            + ' --outcome_name=\'' + args.group_name_readable + '\' --features=' + feature_set
        experiment_str     += this_experiment_str + '\n'
    
    this_script_call_str = 'python3 extract_procedure_groupings.py --include=\'' + args.include \
                         + '\' --exclude=\'' + args.exclude + '\' --group_name=' + args.group_name \
                         + ' --group_name_readable=\'' + args.group_name_readable + '\''
    
    output_dir = 'procedure_groups/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_dir + args.group_name + '.txt', 'w') as f:
        f.write(data_extraction_str + experiment_str + output_str + this_script_call_str)

if __name__ == '__main__':

    parser = create_parser()
    
    args = parser.parse_args()
    assert len(args.include) > 0
    assert len(args.group_name) > 0
    assert ' ' not in args.group_name
    assert len(args.group_name_readable) > 0
    included_list     = args.include.split('%')
    if len(args.exclude) > 0:
        excluded_list = args.exclude.split('%')
    else:
        excluded_list = []
    
    extract_procedure_group(included_list,
                            excluded_list,
                            args)