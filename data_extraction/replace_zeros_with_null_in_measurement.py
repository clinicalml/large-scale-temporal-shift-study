'''
Run with argument --create_table=range_low, range_high, or nulls_replacing_zero
Check table created when running range_low is correct before running nulls_replacing_zero
'''
import argparse
import sqlalchemy
import os
import sys
from os.path import dirname, abspath, join

sys.path.append(dirname(dirname(abspath(__file__))))
sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
import config
from db_utils import session_scope

def create_reference_table(direction, 
                           min_ref_count=5,
                           num_characters=15,
                           avg_diff_threshold=5):
    '''
    Create table {measurement_aux_schema}.measurement_{direction}_references
    with the following columns:
    1. concept_id
    2. concept_name,
    3. similar_concept_id: if reference was taken from a similar concept
    4. similar_concept_name
    5. range_{direction}: reference
    Expects cdm.measurement table to be in database
    @param direction: str, low or high
    @param min_ref_count: int, minimum number of times reference has to occur to be used
    @param num_characters: int, number of initial characters required to match between similar concepts
    @param avg_diff_threshold: int, distance between average non-zero values allowed for similar concepts
    @return: None
    '''
    assert direction in {'low', 'high'}
    assert min_ref_count > 0
    assert num_characters > 0
    assert avg_diff_threshold >= 0
    
    with open('sql/create_measurement_references_table.sql', 'r') as f:
        create_table_sql = f.read()
    if direction == 'low':
        reference_order = 'ASC' # take lowest range low if same frequency
    else:
        reference_order = 'DESC' # take highest range high if same frequency
    create_table_sql     = create_table_sql.format(measurement_aux_schema = config.measurement_aux_schema,
                                                   direction              = direction,
                                                   min_ref_count          = min_ref_count,
                                                   num_characters         = num_characters,
                                                   avg_diff_threshold     = avg_diff_threshold,
                                                   reference_order        = reference_order)
    
    engine = config.create_sqlalchemy_engine()
    with session_scope(engine) as session:
        session.execute(create_table_sql)
        session.commit()
    
def create_table_with_nulls_replacing_zero(threshold=5):
    '''
    Create table {config.measurement_aux_schema}.measurement_with_nulls_replacing_zero
    with same columns as measurement table
    Zeros are replaced with null if it seems like an invalid value for that concept, i.e.
    reference range low or average non-zero value (if reference range low not available) is at least threshold
    and concept does not have negative values
    Expects cdm.measurement and {config.measurement_aux_schema}.measurement_low_references
    @param threshold: int, threshold for comparing reference range low or average non-zero value
    @return: None
    '''
    assert threshold > 0
    
    with open('sql/create_measurement_table_with_nulls_replacing_zero.sql', 'r') as f:
        create_table_sql = f.read()
    create_table_sql     = create_table_sql.format(measurement_aux_schema = config.measurement_aux_schema,
                                                   threshold              = threshold)
    
    engine = config.create_sqlalchemy_engine()
    with session_scope(engine) as session:
        session.execute(create_table_sql)
        session.commit()
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=('Create tables with reference range lows and highs for measurements '
                                                  'and measurement table with zeros correctly replaced with nulls.'))
    parser.add_argument('--create_table',
                        action='store',
                        type=str,
                        help='Specify which table to create: range_low, range_high, or nulls_replacing_zero.')
    args = parser.parse_args()
    assert args.create_table in {'range_low', 'range_high', 'nulls_replacing_zero'}
    
    if args.create_table.startswith('range_'):
        create_reference_table(args.create_table[len('range_'):])
    else:
        create_table_with_nulls_replacing_zero()