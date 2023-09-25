import sqlalchemy

# database parameters
cdm_version                 = 6
sql_version                 = 'postgresql'
db_name                     = 'placeholder'
measurement_aux_schema      = 'placeholder'
nonstationarity_schema_name = 'placeholder'

def create_sqlalchemy_engine():
    '''
    Create sqlalchemy engine to connect to database
    @return: engine
    '''
    engine = sqlalchemy.create_engine(sql_version + '://' + db_name,
                                      echo         = False,
                                      connect_args = {"host": '/var/run/postgresql/'})
    return engine

# directory parameters
cache_dir        = 'placeholder'
omop_data_dir    = 'placeholder'
outcome_data_dir = 'placeholder'
experiment_dir   = 'placeholder'
logging_dir      = 'placeholder'
interaction_dir  = 'placeholder'

domain_shift_dir = 'placeholder'