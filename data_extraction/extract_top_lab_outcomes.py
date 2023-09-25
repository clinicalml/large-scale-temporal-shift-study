import os
import sqlalchemy
import sys
from os.path import dirname, abspath, join

sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from db_utils import session_scope

def select_top_lab_outcomes(number_outcomes,
                            outcome_file):
    '''
    Select most frequent abnormal lab outcomes in the cohort
    If outcome_file exists, nothing is done
    @param number_outcomes: int, number of most frequent outcomes to select
    @param outcome_file: str, file to store outcomes to, if exists, nothing is done
    @return: None
    '''
    if os.path.exists(outcome_file):
        return
    assert number_outcomes > 0
    with open(join(dirname(abspath(__file__)), 
                   'sql', 
                   'select_top_abnormal_lab_outcomes.sql'), 'r') as f:
        top_feat_sql = f.read()
    top_feat_sql     = top_feat_sql.format(number_outcomes        = number_outcomes,
                                           schema_name            = config.nonstationarity_schema_name,
                                           measurement_aux_schema = config.measurement_aux_schema)
    engine = config.create_sqlalchemy_engine()
    with session_scope(engine) as session:
        top_feat_result = session.execute(top_feat_sql)
        session.commit()
    output_str = ('concept_id - concept_name - direction - concept_count - unit - '
                  'male_age_0_30_reference - male_age_31_50_reference - '
                  'male_age_51_70_reference - male_age_over_70_reference - '
                  'female_age_0_30_reference - female_age_31_50_reference - '
                  'female_age_51_70_reference - female_age_over_70_reference (last 9 columns may be repeated)')
    prev_concept_id = ''
    prev_direction  = ''
    for row in top_feat_result:
        curr_concept_id = str(row[0])
        curr_direction  = row[2]
        if curr_concept_id == prev_concept_id and curr_direction == prev_direction:
            output_str += ' - ' + ' - '.join(map(str, row[-9:]))
        else:
            output_str += '\n' + ' - '.join(map(str, row))
        prev_concept_id = curr_concept_id
        prev_direction  = curr_direction
    with open(outcome_file, 'w') as f:
        f.write(output_str)
        
if __name__ == '__main__':
    
    number_outcomes = 150
    
    select_top_lab_outcomes(number_outcomes,
                            config.outcome_data_dir + 'top_abnormal_labs_in_cohort.txt')