import os
import sqlalchemy
import sys
from os.path import dirname, abspath, join

sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from db_utils import session_scope

def select_top_condition_outcomes(number_outcomes,
                                  outcome_file):
    '''
    Select most frequent condition outcomes in the cohort
    If outcome_file exists, nothing is done
    @param number_outcomes: int, number of most frequent outcomes to select
    @param outcome_file: str, file to store outcomes to, if exists, nothing is done
    @return: None
    '''
    if os.path.exists(outcome_file):
        return
    assert number_outcomes > 0
    with open('sql/select_top_condition_outcomes.sql', 'r') as f:
        top_feat_sql = f.read()
    top_feat_sql     = top_feat_sql.format(number_outcomes = number_outcomes,
                                           schema_name     = config.nonstationarity_schema_name)
    engine = sqlalchemy.create_engine('postgresql://' + config.db_name,
                                      echo=False,
                                      connect_args = {"host": '/var/run/postgresql/'})
    with session_scope(engine) as session:
        top_feat_result = session.execute(top_feat_sql)
        session.commit()
    output_str = ''
    for row in top_feat_result:
        output_str += str(row[0]) + ' - ' + row[1] + '\n'
    with open(outcome_file, 'w') as f:
        f.write(output_str)
        
if __name__ == '__main__':
    
    number_outcomes = 100
    
    select_top_condition_outcomes(number_outcomes,
                                  config.outcome_data_dir + 'top_conditions_in_cohort.txt')