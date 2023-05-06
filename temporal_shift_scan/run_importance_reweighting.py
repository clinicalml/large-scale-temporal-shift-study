import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import vstack
from os.path import dirname, abspath, join
from datetime import datetime

from load_data_for_nonstationarity_scan import load_outcomes, load_covariates

sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from logging_utils import set_up_logger
from model_utils import eval_predictions, train_logreg, save_logreg

def get_importance_weights(Xs,
                           feature_names,
                           output_file_header,
                           logger):
    '''
    Train a logistic regression to predict which time point each sample is from
    @param Xs: dict mapping str to list of csr matrices, data split to covariate matrix at 2 time points
    @param feature_names: list of str, feature names
    @param output_file_header: str, start of file path to save model
    @logger: logger, for INFO messages
    @return: dict mapping str to np arrays, data split to importance weights at time 0
    '''
    num_features = len(feature_names)
    assert {'train', 'valid'}.issubset(Xs.keys())
    for data_split in Xs:
        assert np.all(np.array([X.shape[1] == num_features for X in Xs[data_split]]))
    
    time_Xs = dict()
    time_Ys = dict()
    for data_split in Xs:
        time_Xs[data_split] = vstack((Xs[data_split][0],
                                      Xs[data_split][1]),
                                     format = 'csr')
        time_Ys[data_split] = np.concatenate((np.zeros(Xs[data_split][0].shape[0]),
                                              np.ones(Xs[data_split][1].shape[0])))
    
    logreg_file_name = output_file_header + '.joblib'
    if os.path.exists(logreg_file_name):
        time_logreg  = joblib.load(logreg_file_name)
    else:
        time_logreg  = train_logreg(time_Xs['train'],
                                    time_Ys['train'],
                                    time_Xs['valid'],
                                    time_Ys['valid'],
                                    logger)

        save_logreg(time_logreg,
                    output_file_header,
                    feature_names,
                    logger)
    
    logger.info('Evaluating logistic regression predicting year on test data')
    test_time_preds = time_logreg.predict_proba(time_Xs['test'])[:,1]
    eval_predictions(time_Ys['test'],
                     test_time_preds,
                     logger)
    
    importance_weights = dict()
    for data_split in Xs:
        time_preds = time_logreg.predict_proba(Xs[data_split][0])[:,1]
        unclipped = np.divide(time_preds, 1 - time_preds)
        importance_weights[data_split] = np.clip(np.divide(time_preds, 1 - time_preds), .01, 10)
        logger.info(data_split + ' importance weights')
        p025 = np.percentile(importance_weights[data_split], 2.5)
        p25  = np.percentile(importance_weights[data_split], 25)
        p50  = np.percentile(importance_weights[data_split], 50)
        p75  = np.percentile(importance_weights[data_split], 75)
        p975 = np.percentile(importance_weights[data_split], 97.5)
        logger.info('2.5th percentile: ' + str(p025))
        logger.info('25th percentile: ' + str(p25))
        logger.info('50th percentile: ' + str(p50))
        logger.info('75th percentile: ' + str(p75))
        logger.info('97.5th percentile: ' + str(p975))
        p0   = np.min(unclipped)
        p100 = np.max(unclipped)
        logger.info('Min: ' + str(p0))
        logger.info('Max: ' + str(p100))
        
        plt.clf()
        plt.rc('font', 
               family = 'serif', 
               size   = 14)
        plt.rc('xtick', 
               labelsize = 12)
        plt.rc('ytick', 
               labelsize = 12)
        fig, ax = plt.subplots(nrows = 1,
                               ncols = 1,
                               figsize = (6.4, 3.2))
        logbins = np.logspace(0.01, 10, 100)
        ax.hist(importance_weights[data_split],
                bins  = logbins,
                range = [0.01, 10])
        ax.set_xlabel('Importance weight')
        ax.set_xscale('log')
        ax.set_ylabel('Frequency')
        ax.set_xlim([.01,10])
        plt.tight_layout()
        plt.savefig(output_file_header + 'importance_weight_dist_' + data_split + '.pdf')
        
    return importance_weights
    
if __name__ == '__main__':
    
    experiment_dir = config.experiment_dir + 'condition_378253_outcomes_from_all_freq100_logreg/'
    data_dir       = config.outcome_data_dir + 'dataset_condition_378253_outcomes/'
    data_header    = data_dir + 'fold0_freq100'
    
    logging_filename = experiment_dir + 'run_importance_reweighting_' \
                     + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.log'
    logger           = set_up_logger('logger_main',
                                     logging_filename)
    
    Xs, feature_names, _ = load_covariates(data_header,
                                           'all',
                                           2,
                                           logger,
                                           starting_year_idx = 2)
    Ys = load_outcomes(data_header,
                       2,
                       logger,
                       starting_year_idx = 2)
    
    importance_weights_file_header = experiment_dir + 'importance_reweighting_logreg_2020_v_2019'
    importance_weights = get_importance_weights(Xs,
                                                feature_names,
                                                importance_weights_file_header,
                                                logger)
    
    weighted_logreg_file_header = experiment_dir + 'importance_reweighted_logreg_2019'
    weighted_logreg_file_name   = weighted_logreg_file_header + '.joblib'
    if os.path.exists(weighted_logreg_file_name):
        importance_weighted_logreg = joblib.load(weighted_logreg_file_name)
    else:
        importance_weighted_logreg = train_logreg(Xs['train'][0],
                                                  Ys['train'][0],
                                                  Xs['valid'][0],
                                                  Ys['valid'][0],
                                                  logger,
                                                  sample_weight_train = importance_weights['train'],
                                                  sample_weight_valid = importance_weights['valid'])
        save_logreg(importance_weighted_logreg,
                    weighted_logreg_file_header,
                    feature_names,
                    logger)

    logger.info('Evaluating importance re-weighted 2019 logistic regression on 2020 test data')
    importance_weighted_pred = importance_weighted_logreg.predict_proba(Xs['test'][1])[:,1]
    eval_predictions(Ys['test'][1],
                     importance_weighted_pred,
                     logger)