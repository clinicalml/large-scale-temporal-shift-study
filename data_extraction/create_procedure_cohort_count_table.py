import sys
from os.path import dirname, abspath, join

sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from db_utils import session_scope

if __name__ == '__main__':
    
    with open('sql/create_procedure_cohort_count_table.sql', 'r') as f:
        procedure_count_sql = f.read()
    procedure_count_sql     = procedure_count_sql.format(schema_name = config.nonstationarity_schema_name)
    
    engine = config.create_sqlalchemy_engine()
    with session_scope(engine) as session:
        proc_results = session.execute(sqlalchemy.text(procedure_count_sql))
        session.commit()