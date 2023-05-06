import os
import sqlalchemy
import sys
import argparse
from os.path import dirname, abspath, join

sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from db_utils import session_scope

def create_lab_reference_table(direction,
                               min_ref_count      = 5,
                               num_characters     = 15,
                               avg_diff_threshold = 5):
    '''
    Create table {nonstationarity_schema_name}.measurement_age_gender_specific_{direction}_references
    Tables will have the following columns:
    1. concept_id
    2. concept_name
    3. age_range: entries range 1: <= 30, range 2: 31 - 50, range 3: 51 - 70, range 4: > 70
    4. gender_source_value: M or F
    5. similar_concept_id: if reference was taken from a similar concept
    6. similar_concept_name
    7. range_{direction}: reference
    8. from_opposite_gender: 1 if reference was taken from opposite gender
    9. from_different_age_range: 1 if reference was taken from another age range - not present in eol_version
    10. from_general_reference: 1 if reference was taken from general population
    @param direction: str, low or high, direction of reference
    @param min_ref_count: int, minimum number of times reference has to occur to be used
    @param num_characters: int, number of initial characters required to match between similar concepts
    @param avg_diff_threshold: int, distance between average non-zero values allowed for similar concepts
    @return: None
    '''
    assert direction in {'low', 'high'}
    
    if direction == 'low':
        reference_order = 'ASC'
    else:
        reference_order = 'DESC'
    
    reference_table_sql_filename = 'create_age_gender_specific_reference_table.sql'
    
    with open(join(dirname(abspath(__file__)), 
                   'sql', 
                   reference_table_sql_filename), 'r') as f:
        create_table_sql = f.read()
    create_table_sql     = create_table_sql.format(direction          = direction,
                                                   reference_order    = reference_order,
                                                   min_ref_count      = min_ref_count,
                                                   num_characters     = num_characters,
                                                   avg_diff_threshold = avg_diff_threshold,
                                                   schema_name        = config.nonstationarity_schema_name)
    
    engine = sqlalchemy.create_engine('postgresql://' + config.db_name,
                                      echo=False,
                                      connect_args = {"host": '/var/run/postgresql/'})
    with session_scope(engine) as session:
        session.execute(create_table_sql)
        session.commit()
        
def standardize_lab_reference_table(direction):
    '''
    Create table {nonstationarity_schema_name}.measurement_age_gender_specific_standardized_{direction}_references
    with the same structure as {nonstationarity_schema_name}.measurement_age_gender_specific_{direction}_references
    and some references replaced by values from clinical sources.
    @param direction: str, low or high, direction of reference
    @return: None
    '''
    assert direction in {'low', 'high'}
    
    reference_table_sql_filename = 'create_standardized_lab_reference_tables.sql'
    with open(join(dirname(abspath(__file__)), 
                   'sql', 
                   reference_table_sql_filename), 'r') as f:
        create_table_sql = f.read()
    create_table_sql     = create_table_sql.format(direction   = direction,
                                                   schema_name = config.nonstationarity_schema_name)
    
    engine = sqlalchemy.create_engine('postgresql://' + config.db_name,
                                      echo=False,
                                      connect_args = {"host": '/var/run/postgresql/'})
    with session_scope(engine) as session:
        session.execute(create_table_sql)
        session.commit()
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Extract reference tables.')
    parser.add_argument('--direction', 
                        action='store', 
                        type=str, 
                        help='Specify whether creating table for low or high reference ranges.')
    parser.add_argument('--standardize_only',
                        action='store_true',
                        default=False,
                        help='Only standardize references table. Assumes references table was already created.')
    
    args = parser.parse_args()
    assert args.direction in {'low', 'high'}
    
    if not args.standardize_only:
        # larger threshold than general reference table since smaller cohort has larger variance
        create_lab_reference_table(args.direction,
                                   avg_diff_threshold = 10)
    
    standardize_lab_reference_table(args.direction)