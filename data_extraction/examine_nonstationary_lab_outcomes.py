import sys
import argparse
import json
import sqlalchemy
from datetime import datetime

from os.path import dirname, abspath, join
sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from logging_utils import set_up_logger
from db_utils import session_scope

def log_lab_and_outcome_frequency(outcome_name,
                                  year,
                                  logger):
    '''
    Log changes in outcome and lab frequency
    @param outcome_name: str, outcome name as written in file names
    @param year: int, will log change from prior year to this year
    @param logger: logger, for INFO messages
    @return: None
    '''
    outcome_freq_filename = config.outcome_data_dir + 'dataset_' + outcome_name + '/' \
                          + outcome_name + '_cohort_size_outcome_freq.json'
    with open(outcome_freq_filename, 'r') as f:
        outcome_frequencies = json.load(f)
    curr_year_idx = year - 2015
    prev_year_idx = curr_year_idx - 1
    logger.info(outcome_name + ' in ' + str(year))
    logger.info('Outcome frequency changed from ' + str(outcome_frequencies['Outcome frequency'][prev_year_idx]) 
                + ' to ' + str(outcome_frequencies['Outcome frequency'][curr_year_idx]))
    curr_year_lab_freq \
        = outcome_frequencies['Cohort with lab'][curr_year_idx]/float(outcome_frequencies['Cohort size'][curr_year_idx])
    prev_year_lab_freq \
        = outcome_frequencies['Cohort with lab'][prev_year_idx]/float(outcome_frequencies['Cohort size'][prev_year_idx])
    logger.info('Lab frequency changed from ' + str(prev_year_lab_freq) + ' to ' + str(curr_year_lab_freq))

def compute_order_and_available_rate(order_rate_sql_results,
                                     cohort_size):
    '''
    Compute rate lab is ordered and available from sql results
    @param order_rate_sql_results: rows, containing value_available and num_measurements
    @param cohort_size: int, number of patients in cohort
    @return: 1. float, order rate per person
             2. float, available rate per order
    '''
    order_counts        = dict()
    for row in order_rate_sql_results:
        order_counts[row['value_available']] = row['num_measurements']
    available_count     = 0
    if 1 in order_counts:
        available_count = order_counts[1]
    order_total         = available_count
    if 0 in order_counts:
        order_total    += order_counts[0]
    order_rate          = order_total/float(cohort_size)
    available_rate      = available_count/float(order_total)
    return order_rate, available_rate
    
def log_order_rates(concept_id,
                    year,
                    logger):
    '''
    Log changes in order rate and measurement availability rate within cohort
    @param concept_id: int, lab concept ID
    @param year: int, will log change from prior year to this year
    @param logger: logger, for INFO messages
    @return: None
    '''
    logger.info('Lab ' + str(concept_id) + ' in ' + str(year))
    with open('sql/select_lab_order_rate.sql', 'r') as f:
        order_rate_sql = f.read()
    curr_order_rate_sql = order_rate_sql.format(schema_name = config.nonstationarity_schema_name,
                                                concept_id  = str(concept_id),
                                                start_date  = str(year) + '-01-01',
                                                end_date    = str(year) + '-12-31')
    prev_order_rate_sql = order_rate_sql.format(schema_name = config.nonstationarity_schema_name,
                                                concept_id = str(concept_id),
                                                start_date = str(year - 1) + '-01-01',
                                                end_date   = str(year - 1) + '-12-31')
    cohort_size_sql = 'SELECT COUNT(DISTINCT person_id) AS count FROM {schema_name}.omop_cohort_1_year;'
    cohort_size_sql = cohort_size_sql.format(schema_name = config.nonstationarity_schema_name)
    engine = sqlalchemy.create_engine('postgresql://' + config.db_name,
                                      echo         = False,
                                      connect_args = {"host": '/var/run/postgresql/'})
    with session_scope(engine) as session:
        curr_order_rate_results = session.execute(sqlalchemy.text(curr_order_rate_sql))
        session.commit()
        prev_order_rate_results = session.execute(sqlalchemy.text(prev_order_rate_sql))
        session.commit()
        cohort_size_results     = session.execute(sqlalchemy.text(cohort_size_sql)).fetchone()
        session.commit()
    cohort_size = int(cohort_size_results['count'])
    curr_order_rate, curr_available_rate = compute_order_and_available_rate(curr_order_rate_results,
                                                                            cohort_size)
    prev_order_rate, prev_available_rate = compute_order_and_available_rate(prev_order_rate_results,
                                                                            cohort_size)
    logger.info('Order rate changed from ' + str(prev_order_rate) + ' to ' + str(curr_order_rate))
    logger.info('Measurement available rate changed from ' + str(prev_available_rate) + ' to ' + str(curr_available_rate))
    
def create_parser():
    '''
    Create an argument parser
    @return: argparse ArgumentParser
    '''
    parser = argparse.ArgumentParser(description = ('Examine non-stationary lab outcomes'))
    parser.add_argument('--order_rates',
                        action  = 'store_true',
                        default = False,
                        help    = 'Specify whether to explore outcome frequencies or order rates.')
    return parser
    
if __name__ == '__main__':
    
    parser = create_parser()
    args   = parser.parse_args()
    
    logging_filename    = config.logging_dir + 'examine_nonstationary_lab_outcomes_' \
                        + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.log'
    logger              = set_up_logger('logger_main',
                                        logging_filename)
    logger.info('python3 examine_nonstationary_lab_outcomes.py --order_rates=' + str(args.order_rates))
    
    if args.order_rates:
        lab_concepts = {2016: [3000034,
                               3019170],
                        2017: [3002030,
                               3044491],
                        2018: [3002030]}
        for year in range(2016, 2019):
            for concept_id in lab_concepts[year]:
                log_order_rates(concept_id,
                                year,
                                logger)
    else:
        nonstationary_lab_outcomes = {2016: ['lab_3000034_high_outcomes',
                                             'lab_3009744_low_outcomes',
                                             'lab_3019170_high_outcomes',
                                             'lab_3019170_low_outcomes',
                                             'lab_3020509_high_outcomes',
                                             'lab_3037072_low_outcomes',
                                             'lab_3044491_high_outcomes',
                                             'lab_40765040_low_outcomes'],
                                      2017: ['lab_3002030_low_outcomes',
                                             'lab_3019897_low_outcomes',
                                             'lab_3044491_high_outcomes'],
                                      2018: ['lab_3002030_low_outcomes',
                                             'lab_3015632_low_outcomes',
                                             'lab_3019897_low_outcomes',
                                             'lab_3020509_high_outcomes',
                                             'lab_3044491_high_outcomes'],
                                      2019: ['lab_3020509_high_outcomes',
                                             'lab_3037072_low_outcomes']}

        for year in range(2016, 2020):
            for outcome_name in nonstationary_lab_outcomes[year]:
                log_lab_and_outcome_frequency(outcome_name,
                                              year,
                                              logger)