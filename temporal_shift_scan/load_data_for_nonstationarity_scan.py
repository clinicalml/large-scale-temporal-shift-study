import sys
import os
import numpy as np
import gc
import json
import math
import time
from copy import deepcopy
from itertools import product
from collections import defaultdict
from scipy.sparse import vstack, hstack, csr_matrix, csc_matrix
from os.path import dirname, abspath, join
sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from h5py_utils import load_data_from_h5py, load_sparse_matrix_from_h5py

def load_outcomes(dataset_file_header,
                  num_years,
                  logger,
                  starting_year_idx        = 0,
                  test_dataset_file_header = None):
    '''
    Load outcomes into np arrays
    @param dataset_file_header: str, path to start of data files
    @param num_years: int, number of years of data to load
    @param logger: logger, for INFO messages
    @param starting_year_idx: int, index of year to start loading data
    @param test_dataset_file_header: str, path to start of test data files if different from train and valid, otherwise None
    @return: dictionary str to list of np arrays, str is data split, list over years, np arrays contain outcomes
    '''
    data_splits        = ['train', 'valid', 'test']
    dataset_file_headers    = {data_split: dataset_file_header
                               for data_split in data_splits}
    if test_dataset_file_header is not None:
        dataset_file_headers['test'] = test_dataset_file_header
    year_idxs          = range(starting_year_idx, starting_year_idx + num_years)
    outcomes_file_dict = {data_split: [dataset_file_headers[data_split] + '_outcomes_' + data_split 
                                       + '_year' + str(year_idx) + '.hf5'
                                       for year_idx in year_idxs]
                          for data_split in data_splits}
    Y_all_years_dict   = {data_split: [] for data_split in data_splits}
    for data_split, year_idx in product(data_splits, range(num_years)):
        Y_all_years_dict[data_split].append(load_data_from_h5py(outcomes_file_dict[data_split][year_idx])['outcomes'].flatten())
        logger.info(data_split + ' year ' + str(year_idxs[year_idx]) + ': ' 
                    + str(int(np.sum(Y_all_years_dict[data_split][year_idx])))
                    + ' of ' + str(len(Y_all_years_dict[data_split][year_idx])) + ' samples have outcome 1')
    return Y_all_years_dict
    
def load_person_id_to_sample_indices_mappings(dataset_file_header,
                                              num_years,
                                              starting_year_idx        = 0,
                                              test_dataset_file_header = None):
    '''
    Load person ID to sample indices dictionaries
    @param dataset_file_header: str, path to start of data files
    @param num_years: int, number of years of data to load
    @param starting_year_idx: int, index of year to start loading data
    @param test_dataset_file_header: str, path to start of test data files if different from train and valid, otherwise None
    @return: dict, str to list of defaultdict mapping int to list of ints, str is data split, list over years, 
             inner dict maps person ID to sample indices
    '''
    data_splits                      = ['train', 'valid', 'test']
    dataset_file_headers             = {data_split: dataset_file_header
                                        for data_split in data_splits}
    if test_dataset_file_header is not None:
        dataset_file_headers['test'] = test_dataset_file_header
    year_idxs                        = range(starting_year_idx, starting_year_idx + num_years)
    person_ids_file_dict             = {data_split: [dataset_file_headers[data_split] + '_person_ids_' + data_split 
                                                     + '_year' + str(year_idx) + '.json'
                                                     for year_idx in year_idxs]
                                        for data_split in data_splits}
    person_ids_all_years_dict        = {data_split: [] for data_split in data_splits}
    for data_split, year_idx in product(data_splits, range(num_years)):
        with open(person_ids_file_dict[data_split][year_idx], 'r') as f:
            person_id_to_sample_idx_dict = json.load(f)
        # json saves keys as str, so convert to int
        person_id_to_sample_idx_dict = defaultdict(list, {int(person_id): person_id_to_sample_idx_dict[person_id]
                                                          for person_id in person_id_to_sample_idx_dict})
        person_ids_all_years_dict[data_split].append(person_id_to_sample_idx_dict)
    return person_ids_all_years_dict
    
def load_covariates(dataset_file_header,
                    feature_set,
                    num_years,
                    logger,
                    scale_age_back           = False,
                    starting_year_idx        = 0,
                    test_dataset_file_header = None):
    '''
    Load covariates into csr matrices
    @param dataset_file_header: str, path to start of data files
    @param feature_set: str, name of covariate set
    @param num_years: int, number of years of data to load
    @param logger: logger, for INFO messages
    @param scale_age_back: bool, whether to scale age back to original range if scaling params available
    @param starting_year_idx: int, index of year to start loading data
    @param test_dataset_file_header: str, path to start of test data files if different from train and valid, otherwise None
    @return: 1. dictionary str to list of csr matrices, str is data split, list over years, csr matrices contain covariates
             2. list of str, feature names
             3. bool, whether age was scaled back to original range
    '''
    assert feature_set in {'cond_proc', 'labs', 'drugs', 'all'}
    assert num_years > 0
    
    data_splits             = ['train', 'valid', 'test']
    dataset_file_headers    = {data_split: dataset_file_header
                               for data_split in data_splits}
    if test_dataset_file_header is not None:
        dataset_file_headers['test'] = test_dataset_file_header
    feature_types           = ['age', 'general', 'labs', 'conditions', 'procedures', 'drugs', 'specialties']
    year_idxs               = range(starting_year_idx, starting_year_idx + num_years)
    features_file_dict      = {feat_type: {data_split: [dataset_file_headers[data_split] + '_' + feat_type + '_' + data_split 
                                                        + '_year' + str(year_idx) + '.hf5'
                                                        for year_idx in year_idxs]
                                           for data_split in data_splits}
                               for feat_type in feature_types}
    feature_names_file_dict = {feat_type: dataset_file_headers['test'] + '_'  + feat_type + '_feature_names.json'
                               for feat_type in feature_types}
    X_all_years_dict = {data_split: [[] for year_idx in year_idxs] 
                        for data_split in data_splits}
    
    this_start_time = time.time()
    age_scaled_back = False
    if scale_age_back:
        age_scaler_file_name   = dataset_file_header + '_age_scaling_params.txt'
        if os.path.exists(age_scaler_file_name):
            age_scaled_back    = True
            logger.info('Parameters for re-scaling age available at ' + age_scaler_file_name)
            with open(age_scaler_file_name, 'r') as f:
                age_params     = f.read().strip().split(',')
            age_mean           = float(age_params[0])
            age_std            = float(age_params[1])
        else:
            logger.info('Parameters for re-scaling age are not available at ' + age_scaler_file_name)
    for data_split, year_idx in product(data_splits, range(num_years)):
        age_data = load_data_from_h5py(features_file_dict['age'][data_split][year_idx])['age']
        if age_scaled_back:
            age_data = age_data * age_std + age_mean
        X_all_years_dict[data_split][year_idx].append(csr_matrix(age_data))
    gc.collect()
    feature_names = ['age']
    logger.info('Time to load age data: ' + str(time.time() - this_start_time) + ' seconds')

    this_start_time = time.time()
    for data_split, year_idx in product(data_splits, range(num_years)):
        X_all_years_dict[data_split][year_idx].append(load_sparse_matrix_from_h5py(features_file_dict['general']
                                                                                   [data_split][year_idx]))
    with open(feature_names_file_dict['general'], 'r') as f:
        feature_names.extend(json.load(f))
    gc.collect()
    logger.info('Time to load general data: ' + str(time.time() - this_start_time) + ' seconds')

    if feature_set in {'cond_proc', 'all'}:
        this_start_time = time.time()
        for data_split, year_idx in product(data_splits, range(num_years)):
            conditions_file_name = features_file_dict['conditions'][data_split][year_idx]
            X_all_years_dict[data_split][year_idx].append(load_sparse_matrix_from_h5py(conditions_file_name))
            procedures_file_name = features_file_dict['procedures'][data_split][year_idx]
            X_all_years_dict[data_split][year_idx].append(load_sparse_matrix_from_h5py(procedures_file_name))
        with open(feature_names_file_dict['conditions'], 'r') as f:
            feature_names.extend(json.load(f))
        with open(feature_names_file_dict['procedures'], 'r') as f:
            feature_names.extend(json.load(f))
        gc.collect()
        logger.info('Conditions: '  + str(X_all_years_dict[data_split][year_idx][-2].shape[1]) + ' features')
        logger.info('Procedures: '  + str(X_all_years_dict[data_split][year_idx][-1].shape[1]) + ' features')
        logger.info('Time to load conditions and procedures: ' + str(time.time() - this_start_time) + ' seconds')

    if feature_set in {'drugs', 'all'}:
        this_start_time = time.time()
        for data_split, year_idx in product(data_splits, range(num_years)):
            drugs_file_name = features_file_dict['drugs'][data_split][year_idx]
            X_all_years_dict[data_split][year_idx].append(load_sparse_matrix_from_h5py(drugs_file_name))
        with open(feature_names_file_dict['drugs'], 'r') as f:
            feature_names.extend(json.load(f))
        gc.collect()
        logger.info('Drugs: ' + str(X_all_years_dict[data_split][year_idx][-1].shape[1]) + ' features')
        logger.info('Time to load drugs: ' + str(time.time() - this_start_time) + ' seconds')

    if feature_set in {'labs', 'all'}:
        this_start_time = time.time()
        for data_split, year_idx in product(data_splits, range(num_years)):
            labs_file_name = features_file_dict['labs'][data_split][year_idx]
            X_all_years_dict[data_split][year_idx].append(load_sparse_matrix_from_h5py(labs_file_name))
        with open(feature_names_file_dict['labs'], 'r') as f:
            feature_names.extend(json.load(f))
        gc.collect()
        logger.info('Labs: ' + str(X_all_years_dict[data_split][year_idx][-1].shape[1]) + ' features')
        logger.info('Time to load labs: ' + str(time.time() - this_start_time) + ' seconds')

    if feature_set == 'all':
        this_start_time = time.time()
        for data_split, year_idx in product(data_splits, range(num_years)):
            specialties_file_name = features_file_dict['specialties'][data_split][year_idx]
            X_all_years_dict[data_split][year_idx].append(load_sparse_matrix_from_h5py(specialties_file_name))
        with open(feature_names_file_dict['specialties'], 'r') as f:
            feature_names.extend(json.load(f))
        gc.collect()
        logger.info('Specialties: ' + str(X_all_years_dict[data_split][year_idx][-1].shape[1]) + ' features')
        logger.info('Time to load specialties: ' + str(time.time() - this_start_time) + ' seconds')

    this_start_time = time.time()
    X_csr_all_years_dict = {data_split: [hstack(X_all_years_dict[data_split][year_idx],
                                                format='csr') 
                                         for year_idx in range(num_years)]
                            for data_split in data_splits}
    del X_all_years_dict
    logger.info('Time to stack data: ' + str(time.time() - this_start_time) + ' seconds')
    return X_csr_all_years_dict, feature_names, age_scaled_back