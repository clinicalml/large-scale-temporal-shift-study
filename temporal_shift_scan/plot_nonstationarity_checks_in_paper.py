import os
import json
import sys
import time
import math
import joblib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, abspath, join
from datetime import datetime
from sklearn.metrics import roc_auc_score
import multiprocessing as mp
from functools import partial
import multiprocessing_logging

from load_data_for_nonstationarity_scan import (
    load_outcomes, 
    load_covariates, 
    load_person_id_to_sample_indices_mappings
)

sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from logging_utils import set_up_logger
from nonstationarity_scan_metric_dict_utils import convert_str_to_num_in_metrics_dict
from nonstationarity_check_utils import (
    compute_auc_diff_2_models_1_dataset_patient_bootstraps,
    compute_auc_diff_1_model_2_datasets_patient_bootstraps
)
        
def compute_auc_patient_bootstraps(seed,
                                   n_bootstraps,
                                   person_ids_with_outcome,
                                   person_ids_without_outcome,
                                   person_id_to_sample_idxs_dict,
                                   Y,
                                   model_preds):
    '''
    Compute bootstrap estimates of AUC.
    Bootstraps are at the patient-level stratified by outcome.
    @param seed: int, seed for numpy random generator
    @param n_bootstraps: int, number of bootstraps
    @param person_ids_with_outcome: list of ints, person IDs who have outcome
    @param person_ids_without_outcome: list of ints, person IDs who never have outcome
    @param person_id_to_sample_idxs_dict: dict mapping int to list of ints, person ID to sample indices
    @param Y: np array, sample outcomes
    @param model_preds: np array, predictions for samples from model
    @return: np array, bootstrap estimates of AUC differences
    '''
    np.random.seed(seed)
    bootstrap_aucs = np.empty(n_bootstraps)
    for bootstrap_idx in range(n_bootstraps):
        bootstrap_person_ids_with_outcome    = np.random.choice(np.array(person_ids_with_outcome), 
                                                                len(person_ids_with_outcome))
        bootstrap_person_ids_without_outcome = np.random.choice(np.array(person_ids_without_outcome),
                                                                len(person_ids_without_outcome))
        bootstrap_person_ids                 = np.concatenate((bootstrap_person_ids_with_outcome,
                                                               bootstrap_person_ids_without_outcome))
        
        bootstrap_sample_idxs                = np.concatenate([np.array(person_id_to_sample_idxs_dict[int(person_id)],
                                                                        dtype = int)
                                                               for person_id in bootstrap_person_ids],
                                                              dtype = int)
        
        bootstrap_Y           = Y[bootstrap_sample_idxs]
        bootstrap_model_preds = model_preds[bootstrap_sample_idxs]
        bootstrap_auc         = roc_auc_score(bootstrap_Y, bootstrap_model_preds)
        
        bootstrap_aucs[bootstrap_idx] = bootstrap_auc
    
    return bootstrap_aucs
        
def plot_two_outcomes_new(outcomes_to_plot,
                          plot_titles,
                          logger,
                          compute_metrics_only = False,
                          seed                 = 1007):
    '''
    Create 2 subplots (top-bottom) per outcome:
    1. Test AUC with bootstrap std err of current and previous model at each time point
    2. 90% bootstrap confidence interval of AUC difference between 
        a. previous model at previous and current time point
        b. current and previous model at current time point
    x-coordinate for AUC difference in second subplot is between the two AUCs in first subplot
    Plot 2 outcomes side-by-side to format better
    @param outcomes_to_plot: list of str, names of outcome folders in experiment directory
    @param plot_titles: list of str, outcome names for plot titles
    @param logger: logger, for INFO messages
    @param compute_metrics_only: bool, whether to only compute metrics without plotting
    @param seed: int, initial seed for reproducibility from numpy random generator
    @return: None
    '''
    assert len(outcomes_to_plot) <= 2
    assert len(plot_titles)      == len(outcomes_to_plot)
    assert np.all(np.array([(not outcomes_to_plot[i].startswith('condition_'))
                            for i in range(len(outcomes_to_plot))]))
    if not compute_metrics_only:
        assert len(outcomes_to_plot) == 2
    
    np.random.seed(seed)
    summary_dir = 'experiments_selected_with_multiple_hypothesis_testing/' \
                + 'nonstationarity_2outcomes_figure/'
    
    if not compute_metrics_only:
        # create plot
        plt.clf()
        plt.rc('font', 
               family = 'serif', 
               size   = 14)
        plt.rc('xtick', 
               labelsize = 12)
        plt.rc('ytick', 
               labelsize = 12)
        fig, ax = plt.subplots(nrows   = 2,
                               ncols   = 2,
                               figsize = (12.8, 6.4),
                               sharex  = True)
        plt.subplots_adjust(hspace = .1)
    
    for outcome_idx in range(len(outcomes_to_plot)):
        outcome            = outcomes_to_plot[outcome_idx]
        
        outcome_stats_to_plot_filename = config.experiment_dir + summary_dir + outcome + '_metrics_to_plot.json'
        if os.path.exists(outcome_stats_to_plot_filename):
            with open(outcome_stats_to_plot_filename, 'r') as f:
                outcome_stats_to_plot = json.load(f)
            logger.info('Loaded metrics to plot for ' + outcome + ' from ' + outcome_stats_to_plot_filename)
        else:
            logger.info('Computing metrics to plot for ' + outcome)
            num_years = 6
            outcome_stats_to_plot = {'curr_aucs_to_plot'                : [],
                                     'curr_std_errs_to_plot'            : [],
                                     'prev_aucs_to_plot'                : [],
                                     'prev_std_errs_to_plot'            : [],
                                     'our_alg_auc_diffs_to_plot'        : [],
                                     'our_alg_auc_diffs_ci_lbs_to_plot' : [],
                                     'our_alg_auc_diffs_ci_ubs_to_plot' : [],
                                     'baseline_auc_diffs_to_plot'       : [],
                                     'baseline_auc_diffs_ci_lbs_to_plot': [],
                                     'baseline_auc_diffs_ci_ubs_to_plot': []}
            
            # load models
            start_time = time.time()
            models = []
            for year_idx in range(num_years):
                logreg_filename = config.experiment_dir + outcome + '/' + outcome + '_logistic_regression_time' \
                                + str(year_idx) + '.joblib'
                models.append(joblib.load(logreg_filename))

            # load test data
            outcome_ending_idx  = outcome.index('_from_all_freq300_logreg')
            dataset_file_header = config.outcome_data_dir + 'dataset_' + outcome[:outcome_ending_idx] + '/fold0_freq300'
            
            Y = load_outcomes(dataset_file_header,
                              num_years,
                              logger)['test']
            X, _, _ = load_covariates(dataset_file_header,
                                      'all',
                                      num_years,
                                      logger)
            X       = X['test']
            
            person_id_to_sample_idxs_dict = load_person_id_to_sample_indices_mappings(dataset_file_header,
                                                                                      num_years)['test']
            
            # compute person IDs with and without outcome
            person_ids_with_outcome    = []
            person_ids_without_outcome = []
            for year_idx in range(num_years):
                year_person_ids_with_outcome    = []
                year_person_ids_without_outcome = []
                for person_id in person_id_to_sample_idxs_dict[year_idx]:
                    if np.sum(Y[year_idx][person_id_to_sample_idxs_dict[year_idx][person_id]]) > 0:
                        year_person_ids_with_outcome.append(person_id)
                    else:
                        year_person_ids_without_outcome.append(person_id)
                person_ids_with_outcome.append(year_person_ids_with_outcome)
                person_ids_without_outcome.append(year_person_ids_without_outcome)
            
            # compute current and previous model predictions
            curr_model_preds = [models[year_idx].predict_proba(X[year_idx])[:,1]
                                for year_idx in range(num_years)]
            prev_model_preds = [models[year_idx].predict_proba(X[year_idx + 1])[:,1]
                                for year_idx in range(num_years - 1)]
            logger.info('Time to load models and data: ' + str(time.time() - start_time) + ' seconds')
            
            # recompute point estimates
            for year_idx in range(num_years):
                logger.info('Computing metrics for year ' + str(year_idx))
                curr_auc = roc_auc_score(Y[year_idx], curr_model_preds[year_idx])
                outcome_stats_to_plot['curr_aucs_to_plot'].append(curr_auc)
                logger.info('Current AUC: ' + str(outcome_stats_to_plot['curr_aucs_to_plot'][-1]))
                if year_idx == 0:
                    continue
                prev_auc = roc_auc_score(Y[year_idx], prev_model_preds[year_idx - 1])
                outcome_stats_to_plot['prev_aucs_to_plot'].append(prev_auc)
                logger.info('Previous AUC: ' + str(outcome_stats_to_plot['prev_aucs_to_plot'][-1]))
                outcome_stats_to_plot['our_alg_auc_diffs_to_plot'].append(curr_auc - prev_auc)
                logger.info('Our algorithm AUC diff: ' + str(outcome_stats_to_plot['our_alg_auc_diffs_to_plot'][-1]))
                outcome_stats_to_plot['baseline_auc_diffs_to_plot'].append(
                    outcome_stats_to_plot['curr_aucs_to_plot'][-2] - prev_auc)
                logger.info('Baseline AUC diff: ' + str(outcome_stats_to_plot['baseline_auc_diffs_to_plot'][-1]))
            
            n_bootstraps             = 2000
            n_processes              = min(8, mp.cpu_count())
            n_bootstraps_per_process = math.ceil(n_bootstraps/float(n_processes))
            random_seeds             = np.random.randint(10000, size = n_processes*24)
            random_seeds             = [int(seed) for seed in random_seeds]
            seeds_end_idx            = 0

            for year_idx in range(num_years):
                year_stats_to_plot_filename = config.experiment_dir + summary_dir + outcome + '_year' + str(year_idx) \
                                            + '_metrics_to_plot.json'
                
                if os.path.exists(year_stats_to_plot_filename):
                    with open(year_stats_to_plot_filename, 'r') as f:
                        year_stats_to_plot = json.load(f)
                    logger.info('Loaded metrics to plot for ' + outcome + ' in year ' + str(year_idx) + ' from ' 
                                + year_stats_to_plot_filename)
                else:
                    logger.info('Computing metrics for ' + outcome + ' in year ' + str(year_idx))
                    year_stats_to_plot = dict()
                
                    start_time      = time.time()
                    seeds_start_idx = seeds_end_idx
                    seeds_end_idx   = seeds_start_idx + n_processes
                    with mp.get_context('spawn').Pool(processes=n_processes) as pool:
                        curr_auc_bootstraps_per_process \
                            = pool.map(partial(compute_auc_patient_bootstraps, 
                                               n_bootstraps                  = n_bootstraps_per_process,
                                               person_ids_with_outcome       = person_ids_with_outcome[year_idx],
                                               person_ids_without_outcome    = person_ids_without_outcome[year_idx],
                                               person_id_to_sample_idxs_dict = person_id_to_sample_idxs_dict[year_idx],
                                               Y                             = Y[year_idx],
                                               model_preds                   = curr_model_preds[year_idx]),
                                       random_seeds[seeds_start_idx:seeds_end_idx])
                    curr_auc_bootstraps = np.concatenate(curr_auc_bootstraps_per_process)[:n_bootstraps]
                    year_stats_to_plot['curr_std_errs_to_plot'] = np.std(curr_auc_bootstraps)
                    logger.info('Std err for current AUC: ' + str(year_stats_to_plot['curr_std_errs_to_plot']))
                    logger.info('Time to compute std err for current AUC: ' + str(time.time() - start_time) + ' seconds')

                    if year_idx > 0:
                        start_time      = time.time()
                        seeds_start_idx = seeds_end_idx
                        seeds_end_idx   = seeds_start_idx + n_processes
                        with mp.get_context('spawn').Pool(processes=n_processes) as pool:
                            prev_auc_bootstraps_per_process \
                                = pool.map(partial(compute_auc_patient_bootstraps, 
                                                   n_bootstraps                  = n_bootstraps_per_process,
                                                   person_ids_with_outcome       = person_ids_with_outcome[year_idx],
                                                   person_ids_without_outcome    = person_ids_without_outcome[year_idx],
                                                   person_id_to_sample_idxs_dict = person_id_to_sample_idxs_dict[year_idx],
                                                   Y                             = Y[year_idx],
                                                   model_preds                   = prev_model_preds[year_idx - 1]),
                                           random_seeds[seeds_start_idx:seeds_end_idx])
                        prev_auc_bootstraps = np.concatenate(prev_auc_bootstraps_per_process)[:n_bootstraps]
                        year_stats_to_plot['prev_std_errs_to_plot'] = np.std(prev_auc_bootstraps)
                        logger.info('Std err for previous AUC: ' + str(year_stats_to_plot['prev_std_errs_to_plot']))
                        logger.info('Time to compute std err for previous AUC: ' + str(time.time() - start_time) + ' seconds')

                        start_time = time.time()
                        seeds_start_idx = seeds_end_idx
                        seeds_end_idx   = seeds_start_idx + n_processes
                        with mp.get_context('spawn').Pool(processes=n_processes) as pool:
                            our_alg_bootstraps_per_process \
                                = pool.map(partial(compute_auc_diff_2_models_1_dataset_patient_bootstraps, 
                                                   n_bootstraps                  = n_bootstraps_per_process,
                                                   person_ids_with_outcome       = person_ids_with_outcome[year_idx],
                                                   person_ids_without_outcome    = person_ids_without_outcome[year_idx],
                                                   person_id_to_sample_idxs_dict = person_id_to_sample_idxs_dict[year_idx],
                                                   Y                             = Y[year_idx],
                                                   model_1_preds                 = curr_model_preds[year_idx],
                                                   model_2_preds                 = prev_model_preds[year_idx - 1]),
                                           random_seeds[seeds_start_idx:seeds_end_idx])
                        our_alg_bootstraps = np.concatenate(our_alg_bootstraps_per_process)[:n_bootstraps]
                        year_stats_to_plot['our_alg_auc_diffs_ci_lbs_to_plot'] \
                            = 2 * outcome_stats_to_plot['our_alg_auc_diffs_to_plot'][year_idx - 1] \
                            - np.percentile(our_alg_bootstraps, 95)
                        year_stats_to_plot['our_alg_auc_diffs_ci_ubs_to_plot'] \
                            = 2 * outcome_stats_to_plot['our_alg_auc_diffs_to_plot'][year_idx - 1] \
                            - np.percentile(our_alg_bootstraps, 5)
                        logger.info('Our algorithm AUC diff CI: (' 
                                    + str(year_stats_to_plot['our_alg_auc_diffs_ci_lbs_to_plot']) + ', '
                                    + str(year_stats_to_plot['our_alg_auc_diffs_ci_ubs_to_plot']) + ')')
                        logger.info('Time to compute 90% CI for AUC diff in our algorithm: ' 
                                    + str(time.time() - start_time) + ' seconds')

                        start_time      = time.time()
                        seeds_start_idx = seeds_end_idx
                        seeds_end_idx   = seeds_start_idx + n_processes
                        two_year_person_ids  = set(person_id_to_sample_idxs_dict[year_idx - 1].keys()).union(
                            set(person_id_to_sample_idxs_dict[year_idx].keys()))
                        two_year_person_ids_with_outcome    = []
                        two_year_person_ids_without_outcome = []
                        for person_id in two_year_person_ids:
                            if np.sum(Y[year_idx - 1][person_id_to_sample_idxs_dict[year_idx - 1][person_id]]) > 0:
                                two_year_person_ids_with_outcome.append(person_id)
                            elif np.sum(Y[year_idx][person_id_to_sample_idxs_dict[year_idx][person_id]]) > 0:
                                two_year_person_ids_with_outcome.append(person_id)
                            else:
                                two_year_person_ids_without_outcome.append(person_id)
                            
                        with mp.get_context('spawn').Pool(processes=n_processes) as pool:
                            baseline_bootstraps_per_process \
                                = pool.map(partial(compute_auc_diff_1_model_2_datasets_patient_bootstraps,
                                                   n_bootstraps                    = n_bootstraps_per_process,
                                                   person_ids_with_outcome         = two_year_person_ids_with_outcome,
                                                   person_ids_without_outcome      = two_year_person_ids_without_outcome,
                                                   person_id_to_sample_idxs_dict_1 = person_id_to_sample_idxs_dict[year_idx - 1],
                                                   person_id_to_sample_idxs_dict_2 = person_id_to_sample_idxs_dict[year_idx],
                                                   Y_1                             = Y[year_idx - 1],
                                                   Y_2                             = Y[year_idx],
                                                   dataset_1_preds                 = curr_model_preds[year_idx - 1],
                                                   dataset_2_preds                 = prev_model_preds[year_idx - 1]),
                                           random_seeds[seeds_start_idx:seeds_end_idx])
                        baseline_bootstraps = np.concatenate(baseline_bootstraps_per_process)[:n_bootstraps]
                        year_stats_to_plot['baseline_auc_diffs_ci_lbs_to_plot'] \
                            = 2 * outcome_stats_to_plot['baseline_auc_diffs_to_plot'][year_idx - 1] \
                            - np.percentile(baseline_bootstraps, 95)
                        year_stats_to_plot['baseline_auc_diffs_ci_ubs_to_plot'] \
                            = 2 * outcome_stats_to_plot['baseline_auc_diffs_to_plot'][year_idx - 1] \
                            - np.percentile(baseline_bootstraps, 5)
                        logger.info('Baseline AUC diff CI: (' 
                                    + str(year_stats_to_plot['baseline_auc_diffs_ci_lbs_to_plot']) + ', '
                                    + str(year_stats_to_plot['baseline_auc_diffs_ci_ubs_to_plot']) + ')')
                        logger.info('Time to compute 90% CI for AUC diff in baseline: ' 
                                    + str(time.time() - start_time) + ' seconds')
            
                    with open(year_stats_to_plot_filename, 'w') as f:
                        json.dump(year_stats_to_plot, f)
                    logger.info('Saved metrics to plot for ' + outcome + ' in year ' + str(year_idx) + ' to ' 
                                + year_stats_to_plot_filename)
                
                for metric_name in year_stats_to_plot:
                    outcome_stats_to_plot[metric_name].append(year_stats_to_plot[metric_name])
                    
            with open(outcome_stats_to_plot_filename, 'w') as f:
                json.dump(outcome_stats_to_plot, f)
            logger.info('Saved metrics to plot for ' + outcome + ' to ' + outcome_stats_to_plot_filename)
         
        if compute_metrics_only:
            continue
        
        # create first subplot
        ax[0,outcome_idx].errorbar(np.arange(2015, 2021) + 0.1,
                                   outcome_stats_to_plot['curr_aucs_to_plot'],
                                   yerr    = outcome_stats_to_plot['curr_std_errs_to_plot'],
                                   label   = 'Current model',
                                   c       = 'red',
                                   fmt     = 'o',
                                   capsize = 2)
        
        ax[0,outcome_idx].errorbar(np.arange(2016, 2021) - 0.1,
                                   outcome_stats_to_plot['prev_aucs_to_plot'],
                                   yerr    = outcome_stats_to_plot['prev_std_errs_to_plot'],
                                   label   = 'Previous model',
                                   c       = 'blue',
                                   fmt     = 'o',
                                   capsize = 2)
        legend_loc = 'best'
        if outcome_idx == 0:
            legend_loc = 'lower left'
        ax[0,outcome_idx].legend(loc = legend_loc)
        ax[0,outcome_idx].set_title(plot_titles[outcome_idx])
        if outcome_idx == 0:
            ax[0,outcome_idx].set_ylabel('AUC')
        
        # create second subplot
        lower_errs = np.array(outcome_stats_to_plot['our_alg_auc_diffs_to_plot']) \
                   - np.array(outcome_stats_to_plot['our_alg_auc_diffs_ci_lbs_to_plot'])
        upper_errs = np.array(outcome_stats_to_plot['our_alg_auc_diffs_ci_ubs_to_plot']) \
                   - np.array(outcome_stats_to_plot['our_alg_auc_diffs_to_plot'])
        ax[1,outcome_idx].errorbar(np.arange(2016, 2021) - 0.1,
                                   outcome_stats_to_plot['our_alg_auc_diffs_to_plot'],
                                   yerr    = [lower_errs, upper_errs],
                                   label   = 'Our algorithm',
                                   c       = 'purple',
                                   fmt     = 'o',
                                   capsize = 2)
        
        lower_errs = np.array(outcome_stats_to_plot['baseline_auc_diffs_to_plot']) \
                   - np.array(outcome_stats_to_plot['baseline_auc_diffs_ci_lbs_to_plot'])
        upper_errs = np.array(outcome_stats_to_plot['baseline_auc_diffs_ci_ubs_to_plot']) \
                   - np.array(outcome_stats_to_plot['baseline_auc_diffs_to_plot'])
        ax[1,outcome_idx].errorbar(np.arange(2016, 2021) + 0.1,
                                   outcome_stats_to_plot['baseline_auc_diffs_to_plot'],
                                   yerr    = [lower_errs, upper_errs],
                                   label   = 'Baseline',
                                   c       = 'green',
                                   fmt     = 'o',
                                   capsize = 2)
        ax[1,outcome_idx].legend(loc = 'lower left')
        if outcome_idx == 0:
            ax[1,outcome_idx].set_ylabel('AUC difference')
        ax[1,outcome_idx].set_xlabel('Year')
        ax[1,outcome_idx].axhline(0, 
                                  c         = 'black',
                                  linestyle = 'dashed')
        ax[1,outcome_idx].axhline(0.01,
                                  c         = 'orange',
                                  linestyle = 'dashed')
        
    if not compute_metrics_only:
        outcome0_diff_ymin, outcome0_diff_ymax = ax[1,0].get_ylim()
        outcome1_diff_ymin, outcome1_diff_ymax = ax[1,1].get_ylim()
        diff_ymin = min([outcome0_diff_ymin, outcome1_diff_ymin])
        diff_ymax = max([outcome0_diff_ymax, outcome1_diff_ymax])
        ax[1,0].set_ylim([diff_ymin, diff_ymax])
        ax[1,1].set_ylim([diff_ymin, diff_ymax])
        
        plt.tight_layout()
        plot_filename = config.experiment_dir + summary_dir + 'auc_std_errs_diff_cis_2outcomes.pdf'
        plt.savefig(plot_filename)
        logger.info('Saved plot to ' + plot_filename)
    
def create_parser():
    '''
    Create an argument parser
    @return: argparse ArgumentParser
    '''
    parser = argparse.ArgumentParser(description='Compute metrics for and create figure showing non-stationarity in 2 outcomes.')
    parser.add_argument('--compute_metrics_only_for',
                        action  = 'store',
                        default = '',
                        type    = str,
                        help    = ('Specify outcome to compute metrics for. Options: lab_3034426_high, procedure_colonoscopy. '
                                   'If specified, will not create plot. '
                                   'If unspecified, will compute metrics and plot both outcomes.')
                       )
    return parser

if __name__ == '__main__':
    
    mp.set_start_method('spawn', force=True)
    
    parser = create_parser()
    args   = parser.parse_args()
    assert args.compute_metrics_only_for in {'', 'lab_3034426_high', 'procedure_colonoscopy'}
    
    outcomes_to_plot = ['lab_3034426_high_outcomes_from_all_freq300_logreg',
                        'procedure_colonoscopy_outcomes_from_all_freq300_logreg']
    plot_titles      = ['High prothrombin time lab outcome',
                        'Colonoscopy procedure outcome']
    outcome_seeds    = [1007, 4001]
    if args.compute_metrics_only_for == '':
        seed                 = outcome_seeds[0]
        compute_metrics_only = False
    else:
        outcome_idx = -1
        for i in range(len(outcomes_to_plot)):
            if outcomes_to_plot[i].startswith(args.compute_metrics_only_for):
                outcome_idx = i
                break
        assert outcome_idx != -1
        outcomes_to_plot = outcomes_to_plot[outcome_idx:outcome_idx+1]
        plot_titles      = plot_titles[outcome_idx:outcome_idx+1]
        seed             = outcome_seeds[outcome_idx]
        compute_metrics_only = True
    
    summary_dir = 'experiments_selected_with_multiple_hypothesis_testing/' \
                + 'nonstationarity_2outcomes_figure/'
    if not os.path.exists(config.experiment_dir + summary_dir):
        os.makedirs(config.experiment_dir + summary_dir)
    
    logging_filename = config.experiment_dir + summary_dir + 'plot_nonstationarity_checks_in_paper_' \
                     + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.log'
    logger           = set_up_logger('logger_main',
                                     logging_filename)
    multiprocessing_logging.install_mp_handler()
    logger.info('plot_nonstationarity_checks_in_paper.py'
                ' --compute_metrics_only_for=' + args.compute_metrics_only_for)
    
    plot_two_outcomes_new(outcomes_to_plot,
                          plot_titles,
                          logger,
                          compute_metrics_only = compute_metrics_only,
                          seed                 = seed)