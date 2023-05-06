import sys
import argparse
import joblib
from os.path import dirname, abspath, join
from datetime import datetime

from load_data_for_nonstationarity_scan import load_outcomes, load_covariates

sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from logging_utils import set_up_logger
from model_utils import eval_predictions

def create_parser():
    '''
    Create an argument parser
    @return: argparse ArgumentParser
    '''
    parser = argparse.ArgumentParser(description = ('Check test AUCs from domain shift robust feature experiments.'))
    parser.add_argument('--version',
                        action  = 'store',
                        type    = str,
                        default = '365day',
                        help    = ('Specify which set of robust features to check: 365day or drugs.'))
    return parser

if __name__ == '__main__':
    
    parser = create_parser()
    args   = parser.parse_args()
    assert args.version in {'365day', 'drugs'}
    
    if args.version == '365day':
        feat_set      = 'all'
        window_suffix = '_365day_features'
    else:
        feat_set      = 'drugs'
        window_suffix = ''
    
    experiment_dir = config.experiment_dir + 'condition_378253_outcomes_from_' + feat_set + '_freq100_logreg' \
                   + window_suffix + '/'
    logreg_header  = experiment_dir + 'condition_378253_outcomes_from_' + feat_set + '_freq100_logreg' \
                   + window_suffix + '_logistic_regression_time'
    data_dir       = config.outcome_data_dir + 'dataset_condition_378253_outcomes' + window_suffix + '/'
    data_header    = data_dir + 'fold0_freq100'
    
    logging_filename = experiment_dir + 'check_domain_shift_robust_features_test_auc_' \
                     + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.log'
    logger           = set_up_logger('logger_main',
                                     logging_filename)
    logger.info('python3 check_domain_shift_robust_features_test_auc.py --version=' + args.version)
    
    Xs, _, _ = load_covariates(data_header,
                               feat_set,
                               2,
                               logger,
                               starting_year_idx = 2)
    Ys = load_outcomes(data_header,
                       2,
                       logger,
                       starting_year_idx = 2)
    
    model_2019 = joblib.load(logreg_header + '2.joblib')
    model_2020 = joblib.load(logreg_header + '3.joblib')
    
    model_2019_pred_2019 = model_2019.predict_proba(Xs['test'][0])[:,1]
    model_2019_pred_2020 = model_2019.predict_proba(Xs['test'][1])[:,1]
    model_2020_pred_2020 = model_2020.predict_proba(Xs['test'][1])[:,1]
    
    logger.info('Evaluating 2019 model on 2019 test data')
    eval_predictions(Ys['test'][0],
                     model_2019_pred_2019,
                     logger)
    
    logger.info('Evaluating 2019 model on 2020 test data')
    eval_predictions(Ys['test'][1],
                     model_2019_pred_2020,
                     logger)
    
    logger.info('Evaluating 2020 model on 2020 test data')
    eval_predictions(Ys['test'][1],
                     model_2020_pred_2020,
                     logger)