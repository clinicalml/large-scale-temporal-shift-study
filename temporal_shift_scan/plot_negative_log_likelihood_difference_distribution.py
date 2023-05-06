import sys
import os
import joblib
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from load_data_for_nonstationarity_scan import load_covariates, load_outcomes

from os.path import dirname, abspath, join
sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from logging_utils import set_up_logger

def plot_negative_log_likelihood_differences(X,
                                             Y,
                                             model_1,
                                             model_2,
                                             plot_filename,
                                             baseline_model_1 = None):
    '''
    Plot differences between negative log likelihoods of two models (model 1 - model 2)
    If baseline_model_1 specified, plot differences between (baseline model 1 - model 2) in second subplot
    @param X: csr matrix, # samples x # features, covariates
    @param Y: np array, # samples, outcomes
    @param model_1: model with predict_proba function
    @param model_2: model with predict_proba function
    @param plot_filename: str, path to save plot
    @param baseline_model_1: model with predict_proba function
    @return: None
    '''
    assert X.shape[0] == len(Y)
    assert X.shape[1] == model_1.n_features_in_
    assert X.shape[1] == model_2.n_features_in_
    model_1_preds = model_1.predict_proba(X)
    model_1_nll   = -1 * np.multiply(Y, np.log(model_1_preds[:, 1])) - np.multiply(1 - Y, np.log(1 - model_1_preds[:, 0]))
    
    model_2_preds = model_2.predict_proba(X)
    model_2_nll   = -1 * np.multiply(Y, np.log(model_2_preds[:, 1])) - np.multiply(1 - Y, np.log(1 - model_2_preds[:, 0]))
    
    nll_diffs     = model_1_nll - model_2_nll
    
    if baseline_model_1 is not None:
        baseline_model_1_preds = baseline_model_1.predict_proba(X)
        baseline_model_1_nll   = -1 * np.multiply(Y, np.log(baseline_model_1_preds[:, 1])) \
                               - np.multiply(1 - Y, np.log(1 - baseline_model_1_preds[:, 0]))
        baseline_nll_diffs     = baseline_model_1_nll - model_2_nll
        
        nrows      = 2
        fig_height = 6.4
    else:
        nrows      = 1
        fig_height = 3.2
    
    plt.clf()
    plt.rc('font', 
           family = 'serif', 
           size   = 14)
    plt.rc('xtick', 
           labelsize = 12)
    plt.rc('ytick', 
           labelsize = 12)
    
    fig, ax = plt.subplots(nrows   = nrows,
                           ncols   = 1,
                           figsize = (6.4, fig_height),
                           squeeze = False,
                           sharex  = True)
    ax[0,0].hist(nll_diffs,
                 bins  = 40,
                 range = [-.4, .4])
    ax[0,0].axvline(x  = 0,
                    c  = 'black')
    ax[0,0].set_ylabel('Frequency')
    ax[0,0].set_xlim([-.4, .4])
    if baseline_model_1 is not None:
        ax[0,0].set_title('2016 vs 2015 model')
        ax[1,0].hist(baseline_nll_diffs,
                     bins  = 40,
                     range = [-.4, .4])
        ax[1,0].axvline(x  = 0,
                        c  = 'black')
        ax[1,0].set_ylabel('Frequency')
        ax[1,0].set_xlim([-.4, .4])
        ax[1,0].set_title('Fold 0 vs 1 model in 2016')
        
    ax[-1,0].set_xlabel('Negative log likelihood difference')
    plt.tight_layout()
    plt.savefig(plot_filename)
    
def create_parser():
    '''
    Create an argument parser
    @return: argparse ArgumentParser
    '''
    parser = argparse.ArgumentParser(description = 'Plot NLL difference distribution for sub-population labels.')
    parser.add_argument('--baseline',
                        action  = 'store_true',
                        default = False,
                        help    = ('Specify whether to plot NLL difference for baseline (2 folds of data from the same year) '
                                   'instead of difference across 2 years.'))
    parser.add_argument('--together',
                        action  = 'store_true',
                        default = False,
                        help    = ('Specify whether to plot NLL difference across 2 years and across 2 folds.'))
    return parser
    
if __name__ == '__main__':
    
    parser = create_parser()
    args   = parser.parse_args()
    
    experiment_name     = 'procedure_self_care_training_outcomes_from_all_freq300_logreg'
    experiment_dir      = config.experiment_dir + experiment_name + '/'
    if args.baseline or args.together:
        baseline_dir    = config.experiment_dir + experiment_name + '_fold1_using_fold0_features/'
    dataset_file_header = config.outcome_data_dir + 'dataset_procedure_self_care_training_outcomes/fold0_freq300'
    
    logging_filename    = experiment_dir + experiment_name + '_plot_negative_log_likelihood_difference_distribution_' \
                        + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.log'
    logger              = set_up_logger('logger_main',
                                        logging_filename)
    logger.info('python3 plot_negative_log_likelihood_difference_distribution.py'
                + ' --baseline=' + str(args.baseline)
                + ' --together=' + str(args.together))
    
    source_model = joblib.load(experiment_dir + experiment_name + '_logistic_regression_time0.joblib')
    target_model = joblib.load(experiment_dir + experiment_name + '_logistic_regression_time1.joblib')
    
    if args.baseline or args.together:
        baseline_source_model = joblib.load(baseline_dir + experiment_name + '_logistic_regression_time1.joblib')
    else:
        baseline_source_model = None
    if args.baseline:
        source_model          = baseline_source_model
        baseline_source_model = None
    
    Xs, _, _     = load_covariates(dataset_file_header,
                                   feature_set       = 'all',
                                   num_years         = 1,
                                   logger            = logger,
                                   starting_year_idx = 1)
    Ys           = load_outcomes(dataset_file_header,
                                 num_years           = 1,
                                 logger              = logger,
                                 starting_year_idx   = 1)
    
    for data_split in Xs.keys():
        if args.together:
            plot_filename = experiment_dir + experiment_name + '_' + data_split + '_2016_v_2015_nll_diff_2plots.pdf'
        elif args.baseline:
            plot_filename = baseline_dir + experiment_name + '_' + data_split + '_2016_fold0_v_1_nll_diff.pdf'
        else:
            plot_filename = experiment_dir + experiment_name + '_' + data_split + '_2016_v_2015_nll_diff.pdf'
        plot_negative_log_likelihood_differences(Xs[data_split][0],
                                                 Ys[data_split][0],
                                                 source_model,
                                                 target_model,
                                                 plot_filename,
                                                 baseline_source_model)