import os
import numpy as np
import pandas as pd
import time
import sys
import argparse
import gc
from datetime import datetime
from os.path import dirname, abspath, join
from sklearn.feature_selection import chi2
from scipy.sparse import csc_matrix, csr_matrix
from scipy.stats import pearsonr

from load_data_for_nonstationarity_scan import load_outcomes, load_covariates

sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from logging_utils import set_up_logger
from model_utils import train_statsmodels_logreg, save_statsmodels_logreg

def check_for_interpretable_subpopulations(max_depth,
                                           logger):
    '''
    Look for sub-populations that are completely defined in a leaf at depth at most max_depth
    @param max_depth: int, maximum depth of leaf in tree
    @param logger: logger, for INFO messages
    @return: None
    '''
    testing_results_dir = config.experiment_dir + 'experiments_selected_with_multiple_hypothesis_testing/'
    output_file         = testing_results_dir + 'interpretable_subpopulations_depth' + str(max_depth) + '.csv'
    if os.path.exists(output_file):
        logger.info(output_file + ' already exists')
        return
    
    assert max_depth > 0
    logger.info('Looking for sub-populations at depth at most ' + str(max_depth))
    starting_tab_str         = '\t' * (max_depth + 1)
    interpretable_data_cols  = ['Experiment', 'Outcome name', 'Year', 'Line', 
                                'Train leaf size', 'Train leaf miscalibrated size', 
                                'Valid leaf size', 'Valid leaf miscalibrated size', 'Description']
    interpretable_data       = {col: [] for col in interpretable_data_cols}
    
    # load sub-populations with shift
    testing_results_filename = testing_results_dir + 'multiple_hypothesis_testing.csv'
    testing_results_df       = pd.read_csv(testing_results_filename)
    subpop_results_df        = testing_results_df.loc[~pd.isnull(testing_results_df['Region name'])]
    subpop_results_df.drop_duplicates(subset  = ['Experiment', 'Region name'],
                                      inplace = True)

    for subpop_idx in range(len(subpop_results_df)):
        start_time = time.time()
        
        # load sub-population description for that year
        experiment_name              = subpop_results_df['Experiment'].iloc[subpop_idx]
        outcome_name                 = subpop_results_df['Outcome name'].iloc[subpop_idx]
        experiment_specific_dir      = config.experiment_dir + experiment_name + '/'
        subpop_name                  = 'subpopulation_analysis_errors_dectree'
        if experiment_name.startswith('condition_'):
            starting_year            = 2017
        else:
            starting_year            = 2015
        year_idx                     = int(subpop_results_df['Region name'].iloc[subpop_idx][len('in_year_'):len('in_year_')+1])
        year                         = starting_year + year_idx
        subpop_tree_description_file = experiment_specific_dir + subpop_name + '/' + experiment_name + '_' + subpop_name + '_' \
                                     + subpop_name + '_decision_tree_time' + str(year_idx) + '.txt'
        
        current_features     = []
        has_current_features = []
        with open(subpop_tree_description_file, 'r') as f:
            for line_idx, line in enumerate(f):
                # check max depth requirement
                if line.startswith(starting_tab_str):
                    continue
                
                # check for empty line
                if len(line.strip()) == 0:
                    continue
                
                # process first line
                if line_idx == 0:
                    feat_name = line.split(' <')[0]
                    if feat_name == 'age':
                        current_features.append(line.strip())
                    else:
                        current_features.append(feat_name)
                    has_current_features.append(False)
                    continue
                    
                line_without_tabs = line.strip()
                if line_without_tabs.startswith('yes: '):
                    # process lines that start with yes
                    if line_without_tabs.startswith('yes: train: '):
                        # check if this leaf contains a large portion of mis-calibrated samples
                        line_after_header_split = line_without_tabs[len('yes: train: '):].split(', ')
                        valid_counts_split      = line_after_header_split[1][len('valid: '):].split('/')
                        valid_region_count      = float(valid_counts_split[0])
                        valid_region_total      = float(valid_counts_split[1])
                        if valid_region_count > 20 and valid_region_total/valid_region_count > .75:
                            save_region = True
                            for feature in current_features:
                                if 'No matching concept' in feature:
                                    save_region = False
                                    break
                            if save_region:
                                # save region
                                logger.info('Found an interpretable sub-population for ' + experiment_name 
                                            + ' in year ' + str(year) + ' at line ' + str(line_idx))
                                train_counts_split = line_after_header_split[0].split('/')
                                train_region_count = float(train_counts_split[0])
                                train_region_total = float(train_counts_split[1])
                                interpretable_data['Experiment'].append(experiment_name)
                                interpretable_data['Outcome name'].append(outcome_name)
                                interpretable_data['Year'].append(year)
                                interpretable_data['Line'].append(line_idx)
                                interpretable_data['Train leaf size'].append(train_region_total)
                                interpretable_data['Train leaf miscalibrated size'].append(train_region_count)
                                interpretable_data['Valid leaf size'].append(valid_region_total)
                                interpretable_data['Valid leaf miscalibrated size'].append(valid_region_count)
                                description = ''
                                for i in range(len(current_features)):
                                    if not has_current_features[i]:
                                        description += 'no '
                                    description += current_features[i] + ', '
                                description = description[:-2]
                                interpretable_data['Description'].append(description)
                    else:
                        num_starting_tabs      = 0
                        truncating_line        = line
                        while truncating_line.startswith('\t'):
                            num_starting_tabs += 1
                            truncating_line    = truncating_line[1:]
                        if num_starting_tabs  <= max_depth: # otherwise region will not be explored
                            feat_name          = line_without_tabs[len('yes: '):].split(' <')[0]
                            if feat_name == 'age':
                                current_features.append(line_without_tabs[len('yes: '):])
                            else:
                                current_features.append(feat_name)
                            has_current_features.append(False)
                else:
                    # process lines that start with no
                    assert line_without_tabs.startswith('no: ')
                    num_starting_tabs = 0
                    truncating_line   = line
                    while truncating_line.startswith('\t'):
                        num_starting_tabs += 1
                        truncating_line = truncating_line[1:]
                    while len(current_features) > num_starting_tabs:
                        current_features.pop()
                        has_current_features.pop()
                    has_current_features[-1] = True
                    if line_without_tabs.startswith('no: train: '):
                        # check if this leaf contains a large portion of mis-calibrated samples
                        line_after_header_split = line_without_tabs[len('no: train: '):].split(', ')
                        valid_counts_split      = line_after_header_split[1][len('valid: '):].split('/')
                        valid_region_count      = float(valid_counts_split[0])
                        valid_region_total      = float(valid_counts_split[1])
                        if valid_region_total > 20 and valid_region_count/valid_region_total > .75:
                            save_region = True
                            for feature in current_features:
                                if 'No matching concept' in feature:
                                    save_region = False
                                    break
                            if save_region:
                                # save region
                                logger.info('Found an interpretable sub-population for ' + experiment_name 
                                            + ' in year ' + str(year) + ' at line ' + str(line_idx))
                                train_counts_split = line_after_header_split[0].split('/')
                                train_region_count = float(train_counts_split[0])
                                train_region_total = float(train_counts_split[1])
                                interpretable_data['Experiment'].append(experiment_name)
                                interpretable_data['Outcome name'].append(outcome_name)
                                interpretable_data['Year'].append(year)
                                interpretable_data['Line'].append(line_idx)
                                interpretable_data['Train leaf size'].append(train_region_total)
                                interpretable_data['Train leaf miscalibrated size'].append(train_region_count)
                                interpretable_data['Valid leaf size'].append(valid_region_total)
                                interpretable_data['Valid leaf miscalibrated size'].append(valid_region_count)
                                description = ''
                                for i in range(len(current_features)):
                                    if not has_current_features[i]:
                                        description += 'no '
                                    description += current_features[i] + ', '
                                description = description[:-2]
                                interpretable_data['Description'].append(description)
                    else:
                        num_starting_tabs      = 0
                        truncating_line        = line
                        while truncating_line.startswith('\t'):
                            num_starting_tabs += 1
                            truncating_line    = truncating_line[1:]
                        if num_starting_tabs  <= max_depth: # otherwise region will not be explored
                            feat_name          = line_without_tabs[len('no: '):].split(' <')[0]
                            if feat_name == 'age':
                                current_features.append(line_without_tabs[len('no: '):])
                            else:
                                current_features.append(feat_name)
                            has_current_features.append(False)
        logger.info('Time to check subpopulation ' + str(subpop_idx) + ': ' + experiment_name + ': ' 
                    + str(time.time() - start_time) + ' seconds')
    
    interpretable_df = pd.DataFrame(data    = interpretable_data,
                                    columns = interpretable_data_cols)
    interpretable_df.to_csv(output_file,
                            index = False)
    logger.info('Saved interpretable sub-populations to ' + output_file)
    
def check_for_large_coefficient_changes(num_top_coefs,
                                        logger):
    '''
    Look for coefficients that were important only in the previous or current year of a non-stationary set-up
    @param num_top_coefs: int, number of top coefficients to check
    @param logger: logger, for INFO messages
    @return: None
    '''
    testing_results_dir = config.experiment_dir + 'experiments_selected_with_multiple_hypothesis_testing/'
    output_file         = testing_results_dir + 'top' + str(num_top_coefs) + '_coef_changes.csv'
    if os.path.exists(output_file):
        logger.info(output_file + ' already exists')
        return
    
    assert num_top_coefs > 0
    change_df_cols = ['Experiment', 'Outcome name', 'Year', 'Feature', 'Important in current year']
    change_df_data = {col: [] for col in change_df_cols}
    
    # load experiments with shift
    testing_results_filename = testing_results_dir + 'multiple_hypothesis_testing.csv'
    testing_results_df       = pd.read_csv(testing_results_filename)
    testing_results_df.drop_duplicates(subset  = ['Experiment', 'Year'],
                                       inplace = True)
    
    # words to ignore in drug names
    drug_stop_words = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'tablet', 'oral', 'inject', 'mg', 'ml', 'pack', 
                       'week', 'day', 'delay', 'release', 'capsule', 'disintegrat', 'solution', 'medicat', 'patch', 'daily', 
                       'twice', 'hr', 'topical', 'hour', 'lotion', 'extend', 'regular', 'human', 'actuat', 'gel', 'syringe', 
                       'prefill', 'cream', 'chew', 'inert', 'ingredient', 'opthalmic', 'spray', 'nasal', 'month', 'rectal', 
                       'suppository', 'suspension', 'inhal', 'vaccine', 'virus', 'viral', 'live', 'strain', 'toxoid', 
                       'antigen', 'type', 'protein', 'like', 'surface', 'activ', 'dose', 'dosage', 'meter'}
    regex_specials  = ['{', '}', '.', '^', '(' ,')', '[', ']', '$' , '*', '+', '?', '|']
    
    for experiment_idx in range(len(testing_results_df)):
        start_time = time.time()
        
        # load coefficients for that year
        experiment_name         = testing_results_df['Experiment'].iloc[experiment_idx]
        outcome_name            = testing_results_df['Outcome name'].iloc[experiment_idx]
        experiment_specific_dir = config.experiment_dir + experiment_name + '/'
        if experiment_name.startswith('condition_'):
            starting_year       = 2017
        else:
            starting_year       = 2015
        year                    = testing_results_df['Year'].iloc[experiment_idx]
        year_idx                = year - starting_year
        logreg_file_header      = experiment_specific_dir + experiment_name + '_logistic_regression_time'
        
        def extract_feature_name(feat_name):
            '''
            Extract feature name (remove concept ID, concept type, and window size)
            If there are fewer than 3 ' - ' in name, original name is returned
            @param feat_name: str
            @return: str, extracted name
            '''
            feat_name_parts = feat_name.split(' - ')
            if len(feat_name_parts) < 4:
                return feat_name
            return ' - '.join(feat_name_parts[2:-1]).lower()
        
        curr_year_logreg_coef_file          = logreg_file_header + str(year_idx) + '_coefficients.csv'
        curr_year_logreg_df                 = pd.read_csv(curr_year_logreg_coef_file)
        curr_year_logreg_df['Feature name'] = curr_year_logreg_df['Feature'].apply(extract_feature_name)
        prev_year_logreg_coef_file          = logreg_file_header + str(year_idx - 1) + '_coefficients.csv'
        prev_year_logreg_df                 = pd.read_csv(prev_year_logreg_coef_file)
        prev_year_logreg_df['Feature name'] = prev_year_logreg_df['Feature'].apply(extract_feature_name)
        
        def check_for_change(full_feat_name,
                             feat_name,
                             coef_df):
            '''
            Check if feature only has small/negative coefficients in coef_df
            @param full_feat_name: str, feature name with concept ID, type, and window size if applicable
            @param feat_name: str, feature name only
            @param coef_df: pandas DataFrame, contains Feature, Feature name, and Coefficient columns
            @return: bool, whether feature only has small/negative coefficients in coef_df
            '''
            # check the exact same feature in previous year first since this check is quicker
            feat_in_coef_df = coef_df.loc[coef_df['Feature'] == full_feat_name]
            if len(feat_in_coef_df) > 0:
                assert len(feat_in_coef_df) == 1
                coef        = feat_in_coef_df['Coefficient'].iloc[0]
                if coef >= 1e-4:
                    return False
            
            if ' - ' not in full_feat_name:
                # no similar features to check
                return True
            
            # check for similar features in previous year
            feat_name_parts           = full_feat_name.split(' - ')
            if feat_name_parts[1] == 'drug':
                # for drug features, look for similar features based on any important word in name
                words_in_feat_name    = feat_name.split(' ')
                change_found          = True
                for word in words_in_feat_name:
                    if len(word) <= 3:
                        continue
                    skip_word         = False
                    for word_part in drug_stop_words:
                        if word_part in word:
                            skip_word = True
                            break
                    if skip_word:
                        continue
                    word_escaped      = word
                    for regex_special in regex_specials:
                        word_escaped  = word_escaped.replace(regex_special, '\\' + regex_special)
                    feat_in_coef_df   = coef_df.loc[coef_df['Feature name'].str.contains(word_escaped)]
                    if len(feat_in_coef_df) == 0:
                        continue
                    max_coef          = feat_in_coef_df['Coefficient'].max()
                    if max_coef >= 1e-4:
                        return False
                return True
            
            # for non-drug features, look for similar features based on first 30 characters of name
            truncate_length      = min(30, len(feat_name))
            truncated_feat_name  = feat_name[:truncate_length]
            feat_in_coef_df      = coef_df.loc[coef_df['Feature name'].str.startswith(truncated_feat_name)]
            if len(feat_in_coef_df) == 0:
                return True
            max_coef             = feat_in_coef_df['Coefficient'].max()
            if max_coef < 1e-4:
                return True
            return False
        
        for coef_idx in range(num_top_coefs):
            # check this coefficient is positive in current year
            curr_coef      = curr_year_logreg_df['Coefficient'].iloc[coef_idx]
            if curr_coef < 1e-3:
                break
            
            # check if there are no similar features with positive coefficient in previous year
            full_feat_name = curr_year_logreg_df['Feature'].iloc[coef_idx]
            feat_name      = curr_year_logreg_df['Feature name'].iloc[coef_idx]
            change_found   = check_for_change(full_feat_name,
                                              feat_name,
                                              prev_year_logreg_df)
            
            if change_found:
                # record change that was found
                change_df_data['Experiment'].append(experiment_name)
                change_df_data['Outcome name'].append(outcome_name)
                change_df_data['Year'].append(year)
                change_df_data['Feature'].append(feat_name)
                change_df_data['Important in current year'].append(1)
                logger.info('Found a feature that become more important: ' + feat_name + ' for ' + experiment_name 
                            + ' in ' + str(year))
            
        for coef_idx in range(num_top_coefs):
            # check if this coefficient is positive in previous year
            prev_coef      = prev_year_logreg_df['Coefficient'].iloc[coef_idx]
            if prev_coef < 1e-3:
                break
                
            # check if there are no similar features with positive coefficient in current year
            full_feat_name = prev_year_logreg_df['Feature'].iloc[coef_idx]
            feat_name      = prev_year_logreg_df['Feature name'].iloc[coef_idx]
            change_found   = check_for_change(full_feat_name,
                                              feat_name,
                                              curr_year_logreg_df)
            
            if change_found:
                # record change that was found
                change_df_data['Experiment'].append(experiment_name)
                change_df_data['Outcome name'].append(outcome_name)
                change_df_data['Year'].append(year)
                change_df_data['Feature'].append(feat_name)
                change_df_data['Important in current year'].append(0)
                logger.info('Found a feature that became less important: ' + feat_name + ' for ' + experiment_name 
                            + ' in ' + str(year))
       
        logger.info('Time to check coefficient changes for ' + experiment_name + ' in ' + str(year) + ': ' 
                    + str(time.time() - start_time) + ' seconds')
    
    change_df = pd.DataFrame(data    = change_df_data,
                             columns = change_df_cols)
    change_df.to_csv(output_file,
                     index = False)
    logger.info('Saved large coefficient changes to ' + output_file)
    
def select_relevant_features(X,
                             y,
                             num_selected_features,
                             logger):
    '''
    Select features that are relevant to outcome using chi2
    @param X: csr_matrix, covariates, # samples x # features
    @param y: np array, outcomes, # samples
    @param num_selected_features: int, number of features to select, based on largest chi2 statistics
    @param logger: logger, for INFO messages
    @return: np array, selected feature indices
    '''
    
    chi2_stats, _           = chi2(X, y)
    chi2_stats_nan_adjusted = np.where(np.isnan(chi2_stats), -1, chi2_stats)
    chi2_threshold_idx      = np.argpartition(chi2_stats_nan_adjusted, -1 * num_selected_features)[-1 * num_selected_features]
    chi2_threshold          = chi2_stats_nan_adjusted[chi2_threshold_idx]
    if np.isnan(chi2_threshold):
        chi2_threshold      = np.nanmin(chi2_stats)
    assert not np.isnan(chi2_threshold)
    logger.info('chi2 threshold: ' + str(chi2_threshold))
    return np.argwhere(chi2_stats_nan_adjusted >= chi2_threshold)

def select_features_for_statsmodels(X_csr_all_years_dict,
                                    Y_all_years_dict,
                                    feature_names,
                                    logger):
    '''
    Select relevant features without multicollinearity for statsmodels
    @param X_csr_all_years_dict: dict mapping str to list of csr matrices, data split to covariates over 2 years
    @param Y_all_years_dict: dict mapping str to list of np arrays, data split to outcomes over 2 years
    @param feature_names: list of str, names of features
    @param logger: logger, for INFO messages
    @return: 1. dict mapping str to list of csr matrices, data split to selected covariates over 2 years
             2. list of str, names of selected features
    '''
    num_features = len(feature_names)
    for data_split in X_csr_all_years_dict:
        assert len(X_csr_all_years_dict[data_split]) == 2
        assert np.all(np.array([X_csr_all_years_dict[data_split][year_idx].shape[1] == num_features
                                for year_idx in range(2)]))
    start_time = time.time()
    # only keep lab ordered features to reduce collinearity
    def is_feat_related_to_lab_value(feat_name):
        '''
        Check if feature is related to lab value
        @param feat_name: str, feature name
        @return: bool
        '''
        if ' - lab - ' not in feat_name:
            return False
        lab_value_feat_parts = {' - below 25th percentile - ', 
                                ' - 25th to 50th percentile - ',
                                ' - 50th to 75th percentile - ',
                                ' - above 75th percentile - ',
                                ' - increasing - ',
                                ' - decreasing - ',
                                ' - below range - ',
                                ' - in range - ',
                                ' - above range - '}
        for feat_part in lab_value_feat_parts:
            if feat_part in feat_name:
                return True
        return False

    only_lab_ordered_feature_idxs  = []
    only_lab_ordered_feature_names = []
    for feat_idx in range(len(feature_names)):
        if not is_feat_related_to_lab_value(feature_names[feat_idx]):
            only_lab_ordered_feature_idxs.append(feat_idx)
            only_lab_ordered_feature_names.append(feature_names[feat_idx])
    only_lab_ordered_X_csr_all_years_dict = {data_split: [csr_matrix(csc_matrix(X)[:,only_lab_ordered_feature_idxs])
                                                          for X in X_csr_all_years_dict[data_split]]
                                             for data_split in X_csr_all_years_dict}

    # select features to use in statsmodels since it cannot take sparse matrices as input
    this_start_time          = time.time()
    num_general_features     = 20  # always use, remove a prediction month feature so not perfectly collinear
    num_selected_features    = 100 # per year, merge selected from two years
    year0_train_X_csc_matrix = csc_matrix(only_lab_ordered_X_csr_all_years_dict['train'][0])
    year0_train_X_no_general = csr_matrix(year0_train_X_csc_matrix[:,num_general_features:])
    year1_train_X_csc_matrix = csc_matrix(only_lab_ordered_X_csr_all_years_dict['train'][1])
    year1_train_X_no_general = csr_matrix(year1_train_X_csc_matrix[:,num_general_features:])
    year0_selected_feat_idxs = select_relevant_features(year0_train_X_no_general,
                                                        Y_all_years_dict['train'][0],
                                                        num_selected_features,
                                                        logger)
    year1_selected_feat_idxs = select_relevant_features(year1_train_X_no_general,
                                                        Y_all_years_dict['train'][1],
                                                        num_selected_features,
                                                        logger)
    selected_feat_idxs       = np.unique(np.concatenate((year0_selected_feat_idxs, 
                                                         year1_selected_feat_idxs)))
    selected_feat_idxs       = np.concatenate((np.arange(num_general_features - 1),
                                               selected_feat_idxs + num_general_features))
    # remove infrequent features in either train X matrix, general features do not need to be checked
    year0_selected_feat_freqs \
        = np.squeeze(np.asarray(year0_train_X_csc_matrix[:,selected_feat_idxs[num_general_features - 1:]].sum(axis = 0)))
    year1_selected_feat_freqs \
        = np.squeeze(np.asarray(year1_train_X_csc_matrix[:,selected_feat_idxs[num_general_features - 1:]].sum(axis = 0)))
    min_freq                  = 100
    filtered_feat_idxs        = np.argwhere(np.logical_and(year0_selected_feat_freqs > min_freq,
                                                           year1_selected_feat_freqs > min_freq)).flatten() \
                              + num_general_features - 1

    # remove highly correlated features, keep most frequent feature
    feats_to_remove = set()
    for i in range(len(filtered_feat_idxs)):
        if i in feats_to_remove:
            continue
        filtered_idx_i = filtered_feat_idxs[i]
        idx_i          = selected_feat_idxs[filtered_idx_i]
        freq_idx_i     = filtered_idx_i - num_general_features + 1
        for j in range(i + 1, len(filtered_feat_idxs)):
            if j in feats_to_remove:
                continue
            filtered_idx_j = filtered_feat_idxs[j]
            idx_j          = selected_feat_idxs[filtered_idx_j]
            freq_idx_j     = filtered_idx_j - num_general_features + 1
            # if feature frequencies are too different, features are unlikely to be highly correlated
            if np.abs(year0_selected_feat_freqs[freq_idx_i] - year0_selected_feat_freqs[freq_idx_j]) <= 100:
                year0_correlation, _ = pearsonr(year0_train_X_csc_matrix[:,idx_i].toarray().flatten(),
                                                year0_train_X_csc_matrix[:,idx_j].toarray().flatten())
                if year0_correlation > .95:
                    logger.info(only_lab_ordered_feature_names[idx_i]   + ' and ' 
                                + only_lab_ordered_feature_names[idx_j] + ' are highly correlated with '
                                + 'r = ' + str(year0_correlation)  + ' in year 0')
                    if year0_selected_feat_freqs[freq_idx_i] > year0_selected_feat_freqs[freq_idx_j]:
                        feats_to_remove.add(j)
                        logger.info('Removing ' + only_lab_ordered_feature_names[idx_j])
                    else:
                        feats_to_remove.add(i)
                        logger.info('Removing ' + only_lab_ordered_feature_names[idx_i])
                        break
                    continue
            if np.abs(year1_selected_feat_freqs[freq_idx_i] - year1_selected_feat_freqs[freq_idx_j]) <= 100:
                year1_correlation, _ = pearsonr(year1_train_X_csc_matrix[:,idx_i].toarray().flatten(),
                                                year1_train_X_csc_matrix[:,idx_j].toarray().flatten())
                if year1_correlation > .95:
                    logger.info(only_lab_ordered_feature_names[idx_i]   + ' and ' 
                                + only_lab_ordered_feature_names[idx_j] + ' are highly correlated with '
                                + 'r = ' + str(year1_correlation)  + ' in year 1')
                    if year1_selected_feat_freqs[freq_idx_i] > year1_selected_feat_freqs[freq_idx_j]:
                        feats_to_remove.add(j)
                        logger.info('Removing ' + only_lab_ordered_feature_names[idx_i])
                    else:
                        feats_to_remove.add(i)
                        logger.info('Removing ' + only_lab_ordered_feature_names[idx_j])
                        break
    feat_idxs_mask = np.ones(len(filtered_feat_idxs),
                             dtype = bool)
    feat_idxs_mask[list(feats_to_remove)] = False
    selected_feat_idxs        = np.concatenate((np.arange(num_general_features - 1),
                                                selected_feat_idxs[filtered_feat_idxs[feat_idxs_mask]]))

    logger.info('Selected ' + str(len(selected_feat_idxs)) + ' features to use in statsmodels')
    selected_feature_names   = [only_lab_ordered_feature_names[idx] for idx in selected_feat_idxs]
    selected_X_csr_all_years_dict = {data_split: [csr_matrix(csc_matrix(X)[:,selected_feat_idxs])
                                                  for X in only_lab_ordered_X_csr_all_years_dict[data_split]]
                                     for data_split in X_csr_all_years_dict}
    logger.info('Time to select features to use in statsmodels: ' + str(time.time() - this_start_time) + ' seconds')
    return selected_X_csr_all_years_dict, selected_feature_names
    
def check_for_significant_coefficient_changes(logger):
    '''
    Among non-stationary set-ups, look for coefficients that are significantly positive in one year 
    and significantly negative in the other year
    @param logger: logger, for INFO messages
    @return: None
    '''
    testing_results_dir = config.experiment_dir + 'experiments_selected_with_multiple_hypothesis_testing/'
    output_file         = testing_results_dir + 'significant_coefficient_sign_changes.csv'
    if os.path.exists(output_file):
        logger.info(output_file + ' already exists')
        return
    
    # load experiments with shift
    testing_results_filename = testing_results_dir + 'multiple_hypothesis_testing.csv'
    testing_results_df       = pd.read_csv(testing_results_filename)
    testing_results_df       = testing_results_df.loc[np.logical_and.reduce(
        (pd.isnull(testing_results_df['Region name']),
         testing_results_df['Year'] != 2020,
         ~testing_results_df['Outcome name'].str.startswith('eGFR')))]
    
    change_df_cols = ['Experiment', 'Outcome name', 'Year', 'Feature', 'Sign in current year']
    change_df_data = {col: [] for col in change_df_cols}
    
    for experiment_idx in range(len(testing_results_df)):
        start_time = time.time()
        
        # load data
        experiment_name   = testing_results_df['Experiment'].iloc[experiment_idx]
        outcome_file_name = experiment_name[:experiment_name.index('_from_all_')]
        outcome_name      = testing_results_df['Outcome name'].iloc[experiment_idx]
        data_file_header  = config.outcome_data_dir + 'dataset_' + outcome_file_name + '/fold0_freq'
        if outcome_file_name.startswith('condition_'):
            data_file_header += '100'
            starting_year     = 2017
        else:
            data_file_header += '300'
            starting_year     = 2015
        year              = testing_results_df['Year'].iloc[experiment_idx]
        year_idx          = year - starting_year
        logger.info('Checking for significant coefficient sign changes for ' + experiment_name + ' in ' + str(year))
        
        statsmodels_logreg_file_header = config.experiment_dir + experiment_name + '/' + experiment_name \
                                       + '_statsmodels_year' + str(year_idx) + 'v' + str(year_idx - 1) + '_time'
        statsmodels_logreg_year1_file  = statsmodels_logreg_file_header + str(year_idx - 1) + '_coefficients.csv'
        statsmodels_logreg_year2_file  = statsmodels_logreg_file_header + str(year_idx) + '_coefficients.csv'
        if not (os.path.exists(statsmodels_logreg_year1_file) and os.path.exists(statsmodels_logreg_year2_file)):
            X_csr_all_years_dict, feature_names, _ = load_covariates(data_file_header,
                                                                     'all',
                                                                     num_years         = 2,
                                                                     logger            = logger,
                                                                     starting_year_idx = year_idx - 1)
            
            Y_all_years_dict = load_outcomes(data_file_header,
                                             num_years         = 2,
                                             logger            = logger,
                                             starting_year_idx = year_idx - 1)
            logger.info('Time to load data: ' + str(time.time() - start_time) + ' seconds')
            
            selected_X_csr_all_years_dict, selected_feature_names = select_features_for_statsmodels(X_csr_all_years_dict,
                                                                                                    Y_all_years_dict,
                                                                                                    feature_names,
                                                                                                    logger)
        
        # fit statsmodels logistic regressions
        this_start_time = time.time()
        if os.path.exists(statsmodels_logreg_year1_file):
            logger.info('Reading coefficients from ' + statsmodels_logreg_year1_file)
            logreg_year1_summary_df    = pd.read_csv(statsmodels_logreg_year1_file)
        else:
            logreg_year1 = train_statsmodels_logreg(selected_X_csr_all_years_dict['train'][0],
                                                    Y_all_years_dict['train'][0],
                                                    selected_feature_names,
                                                    outcome_name,
                                                    logger)
            logreg_year1_summary_df = save_statsmodels_logreg(logreg_year1,
                                                              statsmodels_logreg_file_header + str(year_idx - 1),
                                                              logger)
            logger.info('Time to learn statsmodels logistic regression for year 0: ' 
                        + str(time.time() - this_start_time) + ' seconds')
        
        this_start_time = time.time()
        if os.path.exists(statsmodels_logreg_year2_file):
            logger.info('Reading coefficients from ' + statsmodels_logreg_year2_file)
            logreg_year2_summary_df   = pd.read_csv(statsmodels_logreg_year2_file)
        else:
            logreg_year2 = train_statsmodels_logreg(selected_X_csr_all_years_dict['train'][1],
                                                    Y_all_years_dict['train'][1],
                                                    selected_feature_names,
                                                    outcome_name,
                                                    logger)
            logreg_year2_summary_df = save_statsmodels_logreg(logreg_year2,
                                                              statsmodels_logreg_file_header + str(year_idx),
                                                              logger)
            logger.info('Time to learn statsmodels logistic regression for year 1: '
                        + str(time.time() - this_start_time) + ' seconds')
        
        # check for significant coefficient sign changes
        this_start_time = time.time()
        logreg_year1_summary_df.dropna(subset  = ['[0.025', '0.975]'],
                                       inplace = True)
        logreg_year1_pos_coef_df  = logreg_year1_summary_df.loc[logreg_year1_summary_df['[0.025'] > 0]
        logreg_year1_neg_coef_df  = logreg_year1_summary_df.loc[logreg_year1_summary_df['0.975]'] < 0]
        logreg_year2_summary_df.dropna(subset  = ['[0.025', '0.975]'],
                                       inplace = True)
        logreg_year2_pos_coef_df  = logreg_year2_summary_df.loc[logreg_year2_summary_df['[0.025'] > 0]
        logreg_year2_neg_coef_df  = logreg_year2_summary_df.loc[logreg_year2_summary_df['0.975]'] < 0]
        logreg_pos_to_neg_coef_df = logreg_year1_pos_coef_df[['Feature']].merge(logreg_year2_neg_coef_df[['Feature']])
        logreg_neg_to_pos_coef_df = logreg_year1_neg_coef_df[['Feature']].merge(logreg_year2_pos_coef_df[['Feature']])
        for feat_idx in range(len(logreg_pos_to_neg_coef_df)):
            feature_name = logreg_pos_to_neg_coef_df['Feature'].iloc[feat_idx]
            logger.info(feature_name + ' for ' + experiment_name + ' in ' + str(year) + ' has a coefficient that '
                        + 'changes from significantly positive to significantly negative')
            change_df_data['Experiment'].append(experiment_name)
            change_df_data['Outcome name'].append(outcome_name)
            change_df_data['Year'].append(year)
            change_df_data['Feature'].append(logreg_pos_to_neg_coef_df['Feature'].iloc[feat_idx])
            change_df_data['Sign in current year'].append('Negative')
        for feat_idx in range(len(logreg_neg_to_pos_coef_df)):
            feature_name = logreg_neg_to_pos_coef_df['Feature'].iloc[feat_idx]
            logger.info(feature_name + ' for ' + experiment_name + ' in ' + str(year) + ' has a coefficient that '
                        + 'changes from significantly negative to significantly positive')
            change_df_data['Experiment'].append(experiment_name)
            change_df_data['Outcome name'].append(outcome_name)
            change_df_data['Year'].append(year)
            change_df_data['Feature'].append(logreg_neg_to_pos_coef_df['Feature'].iloc[feat_idx])
            change_df_data['Sign in current year'].append('Positive')
        logger.info('Time to check for significant coefficient sign changes: ' + str(time.time() - this_start_time) + ' seconds')
        logger.info('Time to check ' + experiment_name + ' in ' + str(year) + ': ' + str(time.time() - start_time) + ' seconds')
    
    change_df = pd.DataFrame(data    = change_df_data,
                             columns = change_df_cols)
    change_df.to_csv(output_file,
                     index = False)
    logger.info('Saved significant coefficient sign changes to ' + output_file)
    
def create_parser():
    '''
    Create an argument parser
    @return: argparse ArgumentParser
    '''
    parser = argparse.ArgumentParser(description = ('Find examples of conditional shift via coefficient changes '
                                                    'or interpretable sub-populations.'))
    parser.add_argument('--interpretable_subpop',
                        action  = 'store_true',
                        default = False,
                        help    = 'Specify whether to check for interpretable sub-populations.')
    parser.add_argument('--max_depth',
                        action  = 'store',
                        type    = int,
                        default = 5,
                        help    = 'Specify maximum depth of region leaf in sub-population tree for easy interpretability.')
    parser.add_argument('--coef_change',
                        action  = 'store_true',
                        default = False,
                        help    = 'Specify whether to check for large coefficient changes.')
    parser.add_argument('--num_top_coefs',
                        action  = 'store',
                        type    = int,
                        default = 5,
                        help    = 'Specify number of top positive coefficients to look for changes in.')
    parser.add_argument('--significant_coef_change',
                        action  = 'store_true',
                        default = False,
                        help    = 'Specify whether to check for significant coefficient changes with statsmodels.')
    return parser
    
if __name__ == '__main__':
    
    parser       = create_parser()
    args         = parser.parse_args()
    
    logging_filename = config.logging_dir + 'find_conditional_shift_examples_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") \
                     + '.log' 
    logger           = set_up_logger('logger_main',
                                     logging_filename)
    logger.info('python3 find_conditional_shift_example.py'
                + ' --interpretable_subpop='    + str(args.interpretable_subpop)
                + ' --max_depth='               + str(args.max_depth)
                + ' --coef_change='             + str(args.coef_change)
                + ' --num_top_coefs='           + str(args.num_top_coefs)
                + ' --significant_coef_change=' + str(args.significant_coef_change))
    
    if args.interpretable_subpop:
        check_for_interpretable_subpopulations(args.max_depth,
                                               logger)
    
    if args.coef_change:
        check_for_large_coefficient_changes(args.num_top_coefs,
                                            logger)
    
    if args.significant_coef_change:
        check_for_significant_coefficient_changes(logger)