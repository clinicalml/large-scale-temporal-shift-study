import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import dirname, abspath, join
from datetime import datetime
from scipy.sparse import csc_matrix

from load_data_for_nonstationarity_scan import load_outcomes, load_covariates

sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from logging_utils import set_up_logger

def compute_feature_and_outcome_frequencies(outcome_name,
                                            feature_names,
                                            year,
                                            logger,
                                            combine_or   = False,
                                            combine_name = None):
    '''
    Compute the following statistics:
    1. Feature frequency in the 2 years
    2. Outcome frequency in the 2 years
    3. Outcome frequency within each feature group in the 2 years
    @param outcome_name: str, name of outcome for file names
    @param feature_names: list of str, feature names to compute statistics for
    @param year: int, non-stationary year, will be compared with previous year
    @param logger: logger, for INFO messages, statistics will be logged
    @param combine_or: bool, feature is combination of any of the feature names
    @param combine_name: str, name of combined feature
    @return: None
    '''
    if combine_or:
        assert combine_name is not None
    logger.info('Examining ' + outcome_name + ' in ' + str(year))
    if outcome_name.startswith('condition_'):
        year_idx = year - 2017
        freq_str = 'freq100'
    else:
        year_idx = year - 2015
        freq_str = 'freq300'
    
    start_time = time.time()
    data_file_header = config.outcome_data_dir + 'dataset_' + outcome_name + '_outcomes/fold0_' + freq_str 
    X_csr_all_years_dict, all_feature_names, _ = load_covariates(data_file_header,
                                                                 'all',
                                                                 num_years         = 2,
                                                                 logger            = logger,
                                                                 starting_year_idx = year_idx - 1)
    X_csc_all_years_dict = {data_split: [csc_matrix(X)
                                         for X in X_csr_all_years_dict[data_split]]
                            for data_split in X_csr_all_years_dict}
    Y_all_years_dict = load_outcomes(data_file_header,
                                     num_years         = 2,
                                     logger            = logger,
                                     starting_year_idx = year_idx - 1)
    logger.info('Time to load data: ' + str(time.time() - start_time) + ' seconds')
    
    start_time = time.time()
    prev_num_samples   = sum([len(Y_all_years_dict[data_split][0])
                              for data_split in Y_all_years_dict])
    curr_num_samples   = sum([len(Y_all_years_dict[data_split][0])
                              for data_split in Y_all_years_dict])
    prev_outcome_count = sum([Y_all_years_dict[data_split][0].sum()
                              for data_split in Y_all_years_dict])
    curr_outcome_count = sum([Y_all_years_dict[data_split][1].sum()
                              for data_split in Y_all_years_dict])
    prev_outcome_freq  = float(prev_outcome_count)/prev_num_samples
    curr_outcome_freq  = float(curr_outcome_count)/curr_num_samples
    logger.info(outcome_name + ' frequency in ' + str(year - 1) + ': ' + str(prev_outcome_count) + ' of ' + str(prev_num_samples) 
                + ', ' + str(prev_outcome_freq))
    logger.info(outcome_name + ' frequency in ' + str(year)     + ': ' + str(curr_outcome_count) + ' of ' + str(curr_num_samples)
                + ', ' + str(curr_outcome_freq))
    
    if combine_or:
        feat_idxs = [all_feature_names.index(feat_name) for feat_name in feature_names]
        samples_with_feat_count                 = np.zeros(2)
        samples_with_feat_with_outcome_count    = np.zeros(2)
        samples_without_feat_with_outcome_count = np.zeros(2)
        
        for data_split in X_csc_all_years_dict:
            for year_idx in range(2):
                sample_idxs_with_feat \
                    = np.nonzero(np.asarray(X_csc_all_years_dict[data_split][year_idx][:,feat_idxs].sum(axis = 1)).flatten())[0]
                samples_with_feat_count[year_idx]    += len(sample_idxs_with_feat)
                sample_idxs_without_feat = np.ones(X_csc_all_years_dict[data_split][year_idx].shape[0],
                                                   dtype = bool)
                sample_idxs_without_feat[sample_idxs_with_feat] = False
                samples_with_feat_with_outcome_count[year_idx] \
                    += Y_all_years_dict[data_split][year_idx][sample_idxs_with_feat].sum()
                samples_without_feat_with_outcome_count[year_idx] \
                    += Y_all_years_dict[data_split][year_idx][sample_idxs_without_feat].sum()
        
        prev_feat_count = samples_with_feat_count[0]
        prev_feat_freq  = float(prev_feat_count)/prev_num_samples
        curr_feat_count = samples_with_feat_count[1]
        curr_feat_freq  = float(curr_feat_count)/curr_num_samples
        
        logger.info(combine_name + ' frequency in ' + str(year - 1) + ': '
                    + str(prev_feat_count) + ' of ' + str(prev_num_samples)
                    + ', ' + str(prev_feat_freq))
        logger.info(combine_name + ' frequency in ' + str(year) + ': '
                    + str(curr_feat_count) + ' of ' + str(curr_num_samples)
                    + ', ' + str(curr_feat_freq))
        
        prev_outcome_freq_with_feat    = float(samples_with_feat_with_outcome_count[0])/prev_feat_count
        curr_outcome_freq_with_feat    = float(samples_with_feat_with_outcome_count[1])/curr_feat_count
        prev_without_feat_count        = prev_num_samples - prev_feat_count
        curr_without_feat_count        = curr_num_samples - curr_feat_count
        prev_outcome_freq_without_feat = float(samples_without_feat_with_outcome_count[0])/prev_without_feat_count
        curr_outcome_freq_without_feat = float(samples_without_feat_with_outcome_count[1])/curr_without_feat_count
        
        logger.info('Among ' + str(prev_feat_count) + ' samples with ' + combine_name
                    + ' in ' + str(year - 1) + ', ' 
                    + str(samples_with_feat_with_outcome_count[0]) + ' have outcome. '
                    + 'Frequency: ' + str(prev_outcome_freq_with_feat))
        logger.info('Among ' + str(curr_feat_count) + ' samples with ' + combine_name 
                    + ' in ' + str(year) + ', ' 
                    + str(samples_with_feat_with_outcome_count[1]) + ' have outcome. '
                    + 'Frequency: ' + str(curr_outcome_freq_with_feat))
        logger.info('Among ' + str(prev_without_feat_count) + ' samples without ' + combine_name 
                    + ' in ' + str(year - 1) + ', '
                    + str(samples_without_feat_with_outcome_count[0]) + ' have outcome. '
                    + 'Frequency: ' + str(prev_outcome_freq_without_feat))
        logger.info('Among ' + str(curr_without_feat_count) + ' samples without ' + combine_name 
                    + ' in ' + str(year) + ', '
                    + str(samples_without_feat_with_outcome_count[1]) + ' have outcome. '
                    + 'Frequency: ' + str(curr_outcome_freq_without_feat))
        
    else:
        for feat_name in feature_names:
            feat_idx        = all_feature_names.index(feat_name)
            prev_feat_count = sum([X_csc_all_years_dict[data_split][0][:,feat_idx].sum()
                                   for data_split in X_csc_all_years_dict])
            curr_feat_count = sum([X_csc_all_years_dict[data_split][1][:,feat_idx].sum()
                                   for data_split in X_csc_all_years_dict])
            prev_feat_freq  = float(prev_feat_count)/prev_num_samples
            curr_feat_freq  = float(curr_feat_count)/curr_num_samples
            logger.info(feat_name + ' frequency in ' + str(year - 1) + ': ' 
                        + str(prev_feat_count) + ' of ' + str(prev_num_samples)
                        + ', ' + str(prev_feat_freq))
            logger.info(feat_name + ' frequency in ' + str(year) + ': ' 
                        + str(curr_feat_count) + ' of ' + str(curr_num_samples)
                        + ', ' + str(curr_feat_freq))

            samples_with_feat_with_outcome_count    = np.zeros(2)
            samples_without_feat_with_outcome_count = np.zeros(2)
            for data_split in X_csc_all_years_dict:
                for year_idx in range(2):
                    sample_idxs_with_feat \
                        = np.nonzero(X_csc_all_years_dict[data_split][year_idx][:,feat_idx].toarray().flatten())[0]
                    samples_with_feat_with_outcome_count[year_idx] \
                        += Y_all_years_dict[data_split][year_idx][sample_idxs_with_feat].sum()
                    sample_idxs_without_feat = np.ones(X_csc_all_years_dict[data_split][year_idx].shape[0],
                                                       dtype = bool)
                    sample_idxs_without_feat[sample_idxs_with_feat] = False
                    samples_without_feat_with_outcome_count[year_idx] \
                        += Y_all_years_dict[data_split][year_idx][sample_idxs_without_feat].sum()

            prev_outcome_freq_with_feat    = float(samples_with_feat_with_outcome_count[0])/prev_feat_count
            curr_outcome_freq_with_feat    = float(samples_with_feat_with_outcome_count[1])/curr_feat_count
            prev_without_feat_count        = prev_num_samples - prev_feat_count
            curr_without_feat_count        = curr_num_samples - curr_feat_count
            prev_outcome_freq_without_feat = float(samples_without_feat_with_outcome_count[0])/prev_without_feat_count
            curr_outcome_freq_without_feat = float(samples_without_feat_with_outcome_count[1])/curr_without_feat_count
            logger.info('Among ' + str(prev_feat_count) + ' samples with ' + feat_name
                        + ' in ' + str(year - 1) + ', ' 
                        + str(samples_with_feat_with_outcome_count[0]) + ' have outcome. '
                        + 'Frequency: ' + str(prev_outcome_freq_with_feat))
            logger.info('Among ' + str(curr_feat_count) + ' samples with ' + feat_name 
                        + ' in ' + str(year) + ', ' 
                        + str(samples_with_feat_with_outcome_count[1]) + ' have outcome. '
                        + 'Frequency: ' + str(curr_outcome_freq_with_feat))
            logger.info('Among ' + str(prev_without_feat_count) + ' samples without ' + feat_name 
                        + ' in ' + str(year - 1) + ', '
                        + str(samples_without_feat_with_outcome_count[0]) + ' have outcome. '
                        + 'Frequency: ' + str(prev_outcome_freq_without_feat))
            logger.info('Among ' + str(curr_without_feat_count) + ' samples without ' + feat_name 
                        + ' in ' + str(year) + ', '
                        + str(samples_without_feat_with_outcome_count[1]) + ' have outcome. '
                        + 'Frequency: ' + str(curr_outcome_freq_without_feat))
        
    logger.info('Time to compute statistics: ' + str(time.time() - start_time) + ' seconds')
    
def plot_age_distribution(age_across_years,
                          year,
                          outcome_name,
                          logger):
    '''
    Create a histogram of age distribution in 2 years
    @param age_across_years: list of np arrays, age for each sample in 2 years
    @param year: int, non-stationary year, will be compared with previous year
    @param outcome_name: str, name of outcome for file names
    @param logger: logger, for INFO messages, statistics will be logged
    @return: None
    '''
    start_time = time.time()
    assert len(age_across_years) == 2
    plt.clf()
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(nrows   = 2,
                           ncols   = 1,
                           sharex  = True,
                           figsize = (6.4, 6.4))
    ax[0].hist(age_across_years[0],
               range = [0, 100],
               bins  = 20,
               label = str(year - 1))
    ax[0].legend()
    ax[0].set_ylabel('Frequency')
    ax[1].hist(age_across_years[1],
               range = [0, 100],
               bins  = 20,
               label = str(year))
    ax[1].legend()
    ax[1].set_ylabel('Frequency')
    ax[1].set_xlabel('Age')
    plt.tight_layout()
    plot_filename = config.outcome_data_dir + 'dataset_' + outcome_name + '_outcomes/' + outcome_name \
                  + '_age_distribution_in_' + str(year - 1) + 'v' + str(year) + '.pdf'
    plt.savefig(plot_filename)
    logger.info('Saved age distribution plot to ' + plot_filename)
    logger.info('Time to plot age distribution: ' + str(time.time() - start_time) + ' seconds')
    
def plot_outcome_frequency_across_age(age_across_years,
                                      Y_across_years,
                                      year,
                                      outcome_name,
                                      outcome_file_name,
                                      logger):
    '''
    Create a bar plot depicting outcome frequency in each age group in 2 years
    Age groups: range 1: <= 30, range 2: 31 - 50, range 3: 51 - 70, range 4: > 70
    @param age_across_years: list of np arrays, age for each sample in 2 years
    @param Y_across_years: list of np arrays, list of outcomes in 2 years
    @param year: int, non-stationary year, will be compared with previous year
    @param outcome_name: str, name of outcome for plot title
    @param outcome_file_name: str, name of outcome for file names
    @param logger: logger, for INFO messages
    @return: None
    '''
    start_time = time.time()
    start_time = time.time()
    assert len(age_across_years) == 2
    assert len(Y_across_years)   == 2
    assert np.all(np.array([len(age_across_years[year_idx]) == len(Y_across_years[year_idx])
                            for year_idx in range(2)]))
    
    plot_df_cols = ['Age', 'Outcome frequency', 'Year']
    age_groups   = ['<= 30', '31 - 50', '51 - 70', '> 70']
    age_outcome_counts_per_year = []
    for year_idx in range(2):
        age_range_1_idxs          = np.argwhere(age_across_years[year_idx] <= 30)
        age_range_1_outcome_count = Y_across_years[year_idx][age_range_1_idxs].sum()
        age_outcome_counts_per_year.append(age_range_1_outcome_count)
        
        age_range_2_idxs          = np.argwhere(np.logical_and(age_across_years[year_idx]  > 30,
                                                               age_across_years[year_idx] <= 50))
        age_range_2_outcome_count = Y_across_years[year_idx][age_range_2_idxs].sum()
        age_outcome_counts_per_year.append(age_range_2_outcome_count)
        
        age_range_3_idxs          = np.argwhere(np.logical_and(age_across_years[year_idx]  > 50,
                                                               age_across_years[year_idx] <= 70))
        age_range_3_outcome_count = Y_across_years[year_idx][age_range_3_idxs].sum()
        age_outcome_counts_per_year.append(age_range_3_outcome_count)
        
        age_range_4_idxs          = np.argwhere(age_across_years[year_idx]  > 70)
        age_range_4_outcome_count = Y_across_years[year_idx][age_range_4_idxs].sum()
        age_outcome_counts_per_year.append(age_range_4_outcome_count)
        
    plot_df = pd.DataFrame(data    = {plot_df_cols[0]: 2 * age_groups,
                                      plot_df_cols[1]: age_outcome_counts_per_year,
                                      plot_df_cols[2]: 4 * [str(year - 1)] + 4 * [str(year)]},
                           columns = plot_df_cols)
    plt.clf()
    plt.rcParams.update({'font.size': 14})
    sns.barplot(data      = plot_df,
                y         = plot_df_cols[0],
                x         = plot_df_cols[1],
                hue       = plot_df_cols[2],
                hue_order = [str(year - 1), str(year)],
                order     = age_groups)
    plt.title(outcome_name)
    plt.tight_layout()
    plot_filename = config.outcome_data_dir + 'dataset_' + outcome_file_name + '_outcomes/' + outcome_file_name \
                  + '_age_groups_outcome_frequency_in_' + str(year - 1) + 'v' + str(year) + '.pdf'
    plt.savefig(plot_filename)
    logger.info('Saved age group outcome frequency plot to ' + plot_filename)
    logger.info('Time to plot age group outcome frequency: ' + str(time.time() - start_time) + ' seconds')
    
def examine_outcome_and_age(outcome_file_name,
                            outcome_name,
                            year,
                            logger):
    '''
    Examine age distribution in 2 years and outcome frequency in age groups in 2 years
    @param outcome_file_name: str, name of outcome for file names
    @param outcome_name: str, name of outcome for plot title
    @param year: int, non-stationary year, will be compared with previous year
    @param logger: logger, for INFO messages
    @return: None
    '''
    logger.info('Examining ' + outcome_name + ' in ' + str(year))
    if outcome_name.startswith('condition_'):
        year_idx = year - 2017
        freq_str = 'freq100'
    else:
        year_idx = year - 2015
        freq_str = 'freq300'
    
    start_time = time.time()
    data_file_header = config.outcome_data_dir + 'dataset_' + outcome_file_name + '_outcomes/fold0_' + freq_str 
    X_csr_all_years_dict, all_feature_names, age_scaled_back = load_covariates(data_file_header,
                                                                               'cond_proc',
                                                                               num_years         = 2,
                                                                               logger            = logger,
                                                                               scale_age_back    = True,
                                                                               starting_year_idx = year_idx - 1)
    assert age_scaled_back
    data_splits = ['train', 'valid', 'test']
    age_across_years   = [np.concatenate([csc_matrix(X_csr_all_years_dict[data_split][year_idx])[:,0].toarray().flatten()
                                          for data_split in data_splits])
                          for year_idx in range(2)]
    Y_all_years_dict = load_outcomes(data_file_header,
                                     num_years         = 2,
                                     logger            = logger,
                                     starting_year_idx = year_idx - 1)
    Y_across_years     = [np.concatenate([Y_all_years_dict[data_split][year_idx]
                                          for data_split in data_splits])
                          for year_idx in range(2)]
    
    logger.info('Time to load data: ' + str(time.time() - start_time) + ' seconds')
    
    plot_age_distribution(age_across_years,
                          year,
                          outcome_file_name,
                          logger)
    
    plot_outcome_frequency_across_age(age_across_years,
                                      Y_across_years,
                                      year,
                                      outcome_name,
                                      outcome_file_name,
                                      logger)
    
if __name__ == '__main__':
    
    logging_filename = config.logging_dir + 'examine_conditional_shifts_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") \
                     + '.log' 
    logger           = set_up_logger('logger_main',
                                     logging_filename)
    logger.info('python3 examine_conditional_shifts.py')
    
    # check for shift due to binary features
    outcome_year_conditional_features \
        = {'procedure_inpatient_consultation': 
           {2019: ['319835 - condition - Congestive heart failure - 30 days',
                   '764123 - condition - Atherosclerosis of coronary artery without angina pectoris - 30 days']},
           'procedure_nursing': 
           {2016: ['40481919 - condition - Coronary atherosclerosis - 30 days', 
                   '441417 - condition - Incoordination - 30 days',
                   ('2514441 - procedure - Critical care, evaluation and management of the critically ill '
                    'or critically injured patient; first 30-74 minutes - 30 days'),
                   '2514464 - procedure - Nursing facility discharge day management; 30 minutes or less - 30 days',
                   '2514465 - procedure - Nursing facility discharge day management; more than 30 minutes - 30 days',
                   '38004450 - specialty - Anesthesiology - 30 days',
                   ('3049187 - lab - Glomerular filtration rate/1.73 sq M.predicted among non-blacks [Volume Rate/Area] '
                    'in Serum, Plasma or Blood by Creatinine-based formula (MDRD) - ordered - 30 days')]},
           'lab_3015632_low': 
           {2018: [('2514408 - procedure - Subsequent hospital care, per day, for the evaluation and management of a patient, '
                    'which requires at least 2 of these 3 key components: An expanded problem focused interval history; '
                    'An expanded problem focused examination; Medical decision making of moder - 30 days')]},
           'lab_3009744_low':
           {2016: ['38004456 - specialty - Internal Medicine - 30 days']},
           'procedure_injection': 
           {2016: ['764123 - condition - Atherosclerosis of coronary artery without angina pectoris - 30 days']},
           'lab_3044491_high':
           {2018: ['45756825 - specialty - Radiology - 30 days']},
           'condition_46273463': 
           {2019: ['White']},
           'procedure_vaccination': 
           {2018: [('36306178 - lab - Glomerular filtration rate/1.73 sq M.predicted among blacks [Volume Rate/Area] '
                    'in Serum, Plasma or Blood by Creatinine-based formula (CKD-EPI) - ordered - 30 days')]}
          }
    
    for outcome_name in outcome_year_conditional_features:
        for year in outcome_year_conditional_features[outcome_name]:
            compute_feature_and_outcome_frequencies(outcome_name,
                                                    outcome_year_conditional_features[outcome_name][year],
                                                    year,
                                                    logger)
    
    # check specifically for nursing outcome with previous nursing facility discharge
    nursing_discharge_features \
        = ['2514464 - procedure - Nursing facility discharge day management; 30 minutes or less - 30 days',
           '2514465 - procedure - Nursing facility discharge day management; more than 30 minutes - 30 days']
    compute_feature_and_outcome_frequencies('procedure_nursing',
                                            nursing_discharge_features,
                                            2016,
                                            logger,
                                            combine_or   = True,
                                            combine_name = 'Nursing facility discharge')
    
    # check for shift relative to age
    outcome_year_to_examine_age = {'lab_3019897_low': [2017]}
    outcome_name_to_examine_age = {'lab_3019897_low': 'Erythrocyte distribution width'}
    
    for outcome_name in outcome_year_to_examine_age:
        for year in outcome_year_to_examine_age[outcome_name]:
            examine_outcome_and_age(outcome_name,
                                    outcome_name_to_examine_age[outcome_name],
                                    year,
                                    logger)