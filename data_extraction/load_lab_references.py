import sqlalchemy
import sys
from os.path import dirname, abspath, join

sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from db_utils import session_scope

def load_reference_tables():
    '''
    Create four reference tables from clinical sources. See sql/load_lab_references.sql for definitions.
    @return: None
    '''
    create_tables_sql_filename = 'create_lab_reference_tables.sql'
    with open(join(dirname(abspath(__file__)), 
                   'sql', 
                   create_tables_sql_filename), 'r') as f:
        create_tables_sql = f.read()
    create_tables_sql     = create_tables_sql.format(measurement_aux_schema = config.measurement_aux_schema)
    
    engine = sqlalchemy.create_engine('postgresql://' + config.db_name,
                                      echo=False,
                                      connect_args = {"host": '/var/run/postgresql/'})
    with session_scope(engine) as session:
        session.execute(sqlalchemy.text(create_tables_sql))
        session.commit()
        
    table_name_to_file_name = {'measurement_references_from_sources'              : 'lab_thresholds.csv',
                               'measurement_unit_specific_references_from_sources': 'lab_unit_specific_thresholds.csv',
                               'measurement_units_to_drop'                        : 'lab_units_to_drop.csv',
                               'measurements_out_of_range'                        : 'lab_measurements_out_of_range.csv'}
    
    conn   = engine.raw_connection()
    cursor = conn.cursor()
    for table_name in table_name_to_file_name:
        copy_sql = 'COPY {table_name} FROM STDIN WITH (FORMAT CSV, HEADER)'
        copy_sql = copy_sql.format(table_name = table_name)
        with open(join(dirname(abspath(__file__)), 
                       'lab_references', 
                       table_name_to_file_name[table_name]), 'r') as f:
            cursor.copy_expert(copy_sql, f)
        conn.commit()
    
    index_tables_sql_filename = 'index_lab_reference_tables.sql'
    with open(join(dirname(abspath(__file__)), 
                   'sql', 
                   index_tables_sql_filename), 'r') as f:
        index_tables_sql = f.read()
    index_tables_sql     = index_tables_sql.format(measurement_aux_schema = config.measurement_aux_schema)
    with session_scope(engine) as session:
        session.execute(sqlalchemy.text(index_tables_sql))
        session.commit()
        
if __name__ == '__main__':
    load_reference_tables()