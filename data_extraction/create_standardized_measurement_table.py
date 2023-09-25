import sqlalchemy
import sys
from os.path import dirname, abspath, join

sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from db_utils import session_scope

def create_standardized_measurement_table():
    '''
    Create a standardized measurement table {measurement_aux_schema}.measurement_with_nulls_replacing_zero_drop_nonstandard
    @return: None
    '''
    measurement_table_sql_filename = 'create_standardized_measurement_table.sql'
    with open(join(dirname(abspath(__file__)), 
                   'sql', 
                   measurement_table_sql_filename), 'r') as f:
        measurement_table_sql = f.read()
    measurement_table_sql     = measurement_table_sql.format(measurement_aux_schema = config.measurement_aux_schema)
    
    engine = sqlalchemy.create_engine('postgresql://' + config.db_name,
                                      echo=False,
                                      connect_args = {"host": '/var/run/postgresql/'})
    with session_scope(engine) as session:
        session.execute(measurement_table_sql)
        session.commit()
        
if __name__ == '__main__':
    create_standardized_measurement_table()