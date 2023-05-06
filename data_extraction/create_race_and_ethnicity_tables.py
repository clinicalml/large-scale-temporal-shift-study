import sqlalchemy
import sys
from os.path import dirname, abspath, join

sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from db_utils import session_scope

def create_race_and_ethnicity_tables():
    '''
    Create tables race_concepts and ethnicity_concepts
    @return: None
    '''
    with open('sql/create_race_concept_table.sql', 'r') as f:
        race_sql      = f.read()
    with open('sql/create_ethnicity_concept_table.sql', 'r') as f:
        ethnicity_sql = f.read()
        
    engine = sqlalchemy.create_engine('postgresql://' + config.db_name,
                                      echo         = False,
                                      connect_args = {"host": '/var/run/postgresql/'})
    with session_scope(engine) as session:
        session.execute(race_sql.format(schema_name = config.nonstationarity_schema_name))
        session.execute(ethnicity_sql.format(schema_name = config.nonstationarity_schema_name))
        session.commit()
    
if __name__ == '__main__':
    
    create_race_and_ethnicity_tables()