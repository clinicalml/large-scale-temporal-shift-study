import sqlalchemy
from contextlib import contextmanager
from sqlalchemy.orm import sessionmaker
from datetime import datetime

@contextmanager
def session_scope(engine):
    '''
    Create a sqlalchemy session and close it when finished
    @param engine: sqlalchemy engine
    @return: None
    '''
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()
        
def check_date_format(date_str):
    '''
    Check if date_str follows YYYY-MM-DD format
    Raises ValueError if does not follow format
    @param date_str: str
    @return None
    '''
    try:
        datetime.strptime(date_text, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Incorrect data format, should be YYYY-MM-DD")