import numpy as np
import pandas as pd
import argparse
import time
import gc
from scipy.sparse import csc_matrix, csr_matrix, vstack
from scipy.stats import chi2_contingency
import sys
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from os.path import dirname, abspath, join

from load_data_for_nonstationarity_scan import load_covariates, load_outcomes

sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from logging_utils import set_up_logger
from model_utils import train_logreg, save_logreg
from hypothesis_testing_utils import run_benjamini_hochberg

def examine_domain_shift_in_2020(output_dir,
                                 logger):
    '''
    Identify features that shift between 2019 and 2020 and plot their frequencies
    Features are selected via an L1-regularized logistic regression and hypothesis testing
    If plot dataframe has already been created, do nothing
    @param output_dir: str, path to directory for outputs
    @param logger: logger, for INFO messages
    @return: None
    '''
    plot_filename = output_dir + 'domain_shift_2020_v_2019_plot_features.csv'
    if os.path.exists(plot_filename):
        logger.info('Plot dataframe already exists at ' + plot_filename)
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    hypothesis_test_filename = output_dir + 'domain_shift_selected_tests_2020_v_2019.csv'
    if os.path.exists(hypothesis_test_filename):
        p_value_df = pd.read_csv(hypothesis_test_filename)
        logger.info('Read hypothesis testing results from ' + hypothesis_test_filename)
        n_accepted = p_value_df['Accept'].sum()
        logger.info('Accepted ' + str(n_accepted) + ' hypotheses')
    else:
    
        # Load 2020 and 2019 data
        # Cough outcome shows non-stationarity in 2020
        start_time                             = time.time()
        outcome_specific_data_file_header      = config.outcome_data_dir + 'dataset_condition_254761_outcomes/fold0_freq100'
        X_csr_all_years_dict, feature_names, _ = load_covariates(outcome_specific_data_file_header,
                                                                 'all',
                                                                 4,
                                                                 logger)
        data_splits                            = ['train', 'valid', 'test']
        data_from_2_years                      = {data_split: [csc_matrix(X_csr_all_years_dict[data_split][2 + year_idx])
                                                               for year_idx in range(2)]
                                                  for data_split in data_splits}
        del X_csr_all_years_dict
        gc.collect()
        logger.info('Time to load data: ' + str(time.time() - start_time) + ' seconds')

        # Filter for prediction month between April and December
        start_time            = time.time()
        prediction_months     = ['pred_in_Apr', 'pred_in_May', 'pred_in_Jun', 'pred_in_Jul', 'pred_in_Aug', 
                                 'pred_in_Sep', 'pred_in_Oct', 'pred_in_Nov', 'pred_in_Dec']
        prediction_month_idxs = [feature_names.index(month)
                                 for month in prediction_months]
        idxs_from_months      = {data_split: [np.nonzero(data_from_2_years[data_split]
                                                         [year_idx][:, prediction_month_idxs].toarray().sum(axis = 1))[0]
                                              for year_idx in range(2)]
                                 for data_split in data_splits}
        data_from_months      = {data_split: [csc_matrix(csr_matrix(data_from_2_years[data_split][year_idx])
                                                         [idxs_from_months[data_split][year_idx]])
                                              for year_idx in range(2)]
                                 for data_split in data_splits}
        del data_from_2_years
        del idxs_from_months
        gc.collect()
        logger.info('Time to filter prediction months: ' + str(time.time() - start_time) + ' seconds')

        # Select past 30 day features
        start_time             = time.time()
        past_30_day_feat_idxs  = []
        past_30_day_feat_names = []
        for idx in range(len(feature_names)):
            if feature_names[idx].endswith(' - 30 days'):
                past_30_day_feat_idxs.append(idx)
                past_30_day_feat_names.append(feature_names[idx])
        data_to_compare        = {data_split: [data_from_months[data_split][year_idx][:, past_30_day_feat_idxs]
                                               for year_idx in range(2)]
                                  for data_split in data_splits}
        del data_from_months
        del feature_names
        gc.collect()
        logger.info('Time to select past 30 day features: ' + str(time.time() - start_time) + ' seconds')

        # Learn an L1 logreg for 2020 vs 2019
        logreg_filename = output_dir + 'logreg_2020_v_2019.joblib'
        if os.path.exists(logreg_filename):
            logreg = joblib.load(logreg_filename)
            logger.info('Loaded logistic regression from ' + logreg_filename)
        else:
            start_time = time.time()
            X_data = {data_split: vstack(data_to_compare[data_split],
                                         format = 'csr')
                      for data_split in data_splits}
            Y_data = {data_split: np.concatenate((np.zeros(data_to_compare[data_split][0].shape[0]),
                                                  np.ones(data_to_compare[data_split][1].shape[0])))
                      for data_split in data_splits}
            np.random.seed(4027)
            for data_split in data_splits:
                shuffled_idxs = np.arange(len(Y_data[data_split]))
                np.random.shuffle(shuffled_idxs)
                X_data[data_split] = X_data[data_split][shuffled_idxs]
                Y_data[data_split] = Y_data[data_split][shuffled_idxs]
            # C = 1 has best valid AUC: 0.6060
            # C = 0.1 has valid AUC 0.6058
            # C = 0.01 has valid AUC 0.5973
            # C = 0.001 has valid AUC 0.5730
            # Pick C = 0.01 because within 0.01 of best AUC and sparser features
            logreg = train_logreg(X_data['train'], 
                                  Y_data['train'], 
                                  X_data['valid'], 
                                  Y_data['valid'],
                                  logger,
                                  penalty = 'l1',
                                  Cs = [0.01])
            save_logreg(logreg,
                        logreg_filename[:-1*len('.joblib')],
                        past_30_day_feat_names)
            logger.info('Saved logistic regression to ' + logreg_filename)
            logger.info('Time to learn logistic regression: ' + str(time.time() - start_time) + ' seconds')

        # Gather features with non-zero coefficients
        nonzero_feat_idxs = np.nonzero(logreg.coef_.flatten())[0]
        logger.info(str(len(nonzero_feat_idxs)) + ' features have non-zero coefficients')

        # Compute Fisher test p-values on test set
        start_time         = time.time()
        nonzero_feat_names = []
        nonzero_p_values   = []
        nonzero_props_2019 = []
        nonzero_props_2020 = []
        num_2019_samples   = data_to_compare['test'][0].shape[0]
        logger.info(str(num_2019_samples) + ' test samples in 2019')
        num_2020_samples   = data_to_compare['test'][1].shape[0]
        logger.info(str(num_2020_samples) + ' test samples in 2020')
        for feat_idx in nonzero_feat_idxs:
            # if total validation frequency is below 20, do not test this feature
            valid_frequency        = data_to_compare['valid'][0][:, feat_idx].sum() \
                                   + data_to_compare['valid'][0][:, feat_idx].sum()
            if valid_frequency < 20:
                continue
            feat_frequencies       = np.empty((2, 2),
                                              dtype = int)
            feat_frequencies[0, 1] = data_to_compare['test'][0][:, feat_idx].sum()
            feat_frequencies[0, 0] = num_2019_samples - feat_frequencies[0, 1]
            feat_frequencies[1, 1] = data_to_compare['test'][1][:, feat_idx].sum()
            feat_frequencies[1, 0] = num_2020_samples - feat_frequencies[1, 1]
            nonzero_props_2019.append(feat_frequencies[0, 1]/float(num_2019_samples))
            nonzero_props_2020.append(feat_frequencies[1, 1]/float(num_2020_samples))
            _, p_value, _, _       = chi2_contingency(feat_frequencies)
            nonzero_p_values.append(p_value)
            nonzero_feat_names.append(past_30_day_feat_names[feat_idx])
        logger.info('Ran hypothesis tests for ' + str(len(nonzero_props_2020)) + ' features')
        logger.info('Time to run chi squared contingency tests: ' + str(time.time() - start_time) + ' seconds')

        # Compute Bonferroni correction
        start_time = time.time()
        p_value_df = pd.DataFrame({'Feature': nonzero_feat_names,
                                   'P-value': nonzero_p_values,
                                   '2019 frequency': nonzero_props_2019,
                                   '2020 frequency': nonzero_props_2020})
        p_value_df.to_csv(output_dir + 'domain_shift_selected_tests_2020_v_2019_unsorted.csv',
                          index = False)
        p_value_df = run_benjamini_hochberg(p_value_df)
        p_value_df.to_csv(output_dir + 'domain_shift_selected_tests_2020_v_2019.csv',
                          index = False)
        n_accepted = p_value_df['Accept'].sum()
        logger.info('Accepted ' + str(n_accepted) + ' hypotheses')
        logger.info('Time to run Benjamini-Hochberg: ' + str(time.time() - start_time) + ' seconds')
    
    # Plot frequencies for most common accepted features in 2019 and 2020
    if n_accepted > 0:
        start_time  = time.time()
        accepted_df = p_value_df.loc[p_value_df['Accept'] == 1]
        accepted_df['Frequency sum'] = accepted_df['2019 frequency'] + accepted_df['2020 frequency']
        accepted_df.sort_values(by        = 'Frequency sum',
                                ascending = False,
                                inplace   = True)
        del accepted_df['Frequency sum']
        del accepted_df['Critical value']
        del accepted_df['Accept']
        accepted_df.to_csv(plot_filename,
                           index = False)
        logger.info('Time to write features to plot to ' + plot_filename + ': ' + str(time.time() - start_time) + ' seconds')
        
def plot_features(dataframe_filename,
                  plot_filename,
                  logger):
    '''
    Plot frequent features that shift between 2019 and 2020
    @param dataframe_filename: str, path to csv containing features to plot
    @param plot_filename: str, path to output plot
    @param logger: logger, for INFO messages
    @return: None
    '''
    start_time  = time.time()
    assert os.path.exists(dataframe_filename)
    accepted_df = pd.read_csv(dataframe_filename)
    print(accepted_df)
    logger.info('Read features for plot from ' + dataframe_filename)

    outcome_specific_data_file_header = config.outcome_data_dir + 'dataset_condition_254761_outcomes/fold0_freq100'
    Ys = load_outcomes(outcome_specific_data_file_header,
                       num_years         = 2,
                       logger            = logger,
                       starting_year_idx = 2)
    data_splits = ['train', 'valid', 'test']
    n_2019 = sum([len(Ys[data_split][0]) for data_split in data_splits])
    n_2020 = sum([len(Ys[data_split][1]) for data_split in data_splits])
    
    df_2019 = accepted_df[['Feature', '2019 frequency']]
    df_2020 = accepted_df[['Feature', '2020 frequency']]
    df_2019.rename(columns = {'2019 frequency': 'Frequency'},
                   inplace = True)
    df_2019['Year'] = '2019'
    df_2019['CI']   = 1.96 * np.sqrt(df_2019['Frequency'] * (1 - df_2019['Frequency']) / n_2019)
    df_2020.rename(columns = {'2020 frequency': 'Frequency'},
                   inplace = True)
    df_2020['Year'] = '2020'
    df_2020['CI']   = 1.96 * np.sqrt(df_2020['Frequency'] * (1 - df_2020['Frequency']) / n_2020)

    plot_df = pd.concat((df_2019,
                         df_2020),
                        ignore_index = True)
    print(plot_df)
    plt.clf()
    plt.rc('font', 
           family = 'serif', 
           size   = 14)
    plt.rc('xtick', 
           labelsize = 12)
    plt.rc('ytick', 
           labelsize = 12)
    ax = sns.barplot(data      = plot_df,
                     y         = 'Feature',
                     x         = 'Frequency',
                     hue       = 'Year',
                     hue_order = ['2019', '2020'],
                     order     = accepted_df['Feature'].values.tolist())
    x_coords = [p.get_width() for p in ax.patches]
    y_coords = [p.get_y() + 0.5* p.get_height() for p in ax.patches]
    ax.errorbar(x       = x_coords, 
                y       = y_coords, 
                xerr    = plot_df['CI'], 
                capsize = 0,
                fmt     = 'none',
                ecolor  = 'black')
    ax.set(ylabel = None)
    plt.tight_layout()
    plt.savefig(plot_filename)
    logger.info('Time to plot features: ' + str(time.time() - start_time) + ' seconds')
    
if __name__ == '__main__':
    
    # set up logger
    if not os.path.exists(config.domain_shift_dir):
        os.makedirs(config.domain_shift_dir)
    logging_filename = config.domain_shift_dir + 'domain_shift_' \
                     + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.log'
    logger           = set_up_logger('logger_main',
                                     logging_filename)
    
    parser = argparse.ArgumentParser(description='Examine domain shift in 2020.')
    parser.add_argument('--plot_features', 
                        action  = 'store_true', 
                        default = False, 
                        help    = 'Plot feature frequencies from a cleaned csv')
    args = parser.parse_args()
    logger.info('python3 examine_domain_shift_in_2020.py --plot_features=' + str(args.plot_features))
    
    if args.plot_features:
        clean_plot_data_filename = config.domain_shift_dir + 'domain_shift_2020_v_2019_plot_features_cleaned.csv'
        plot_filename            = config.domain_shift_dir + 'domain_shift_2020_v_2019.pdf'
        plot_features(clean_plot_data_filename,
                      plot_filename,
                      logger)
    else:
        examine_domain_shift_in_2020(config.domain_shift_dir,
                                     logger)