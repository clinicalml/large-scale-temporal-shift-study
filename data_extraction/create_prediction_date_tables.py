import sqlalchemy
import sys
from os.path import dirname, abspath, join

sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from db_utils import session_scope

def write_prediction_dates():
    '''
    Write prediction dates from 2014 to 2020 to text file
    @return: None
    '''
    prediction_dates = []
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    for year in range(2014, 2021):
        for month_idx in range(len(months)):
            prediction_dates.append(str(year) + '-' + months[month_idx] + '-01')

    with open('prediction_dates.txt', 'w') as f:
        f.write('\n'.join(prediction_dates))
    
def create_prediction_date_tables():
    '''
    Create tables prediction_dates_1_year and prediction_dates_3_years
    @return: None
    '''
    with open('sql/load_prediction_date_tables.sql', 'r') as f:
        sql = f.read()
        
    engine = config.create_sqlalchemy_engine()
    with session_scope(engine) as session:
        session.execute(sql.format(schema_name = config.nonstationarity_schema_name))
        session.commit()
    
if __name__ == '__main__':
    
    write_prediction_dates()
    create_prediction_date_tables()