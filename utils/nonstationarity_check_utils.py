import time
import math
import numpy as np
import multiprocessing as mp
from functools import partial
from sklearn.metrics import roc_auc_score

def satisfy_patient_outcome_count_minimum(Y,
                                          person_id_to_sample_idxs_dict,
                                          min_person_freq):
    '''
    Check if at least some number of people have the outcome
    @param Y: np array, binary indicators
    @param person_id_to_sample_idxs_dict: dict mapping int to list of ints, person ID to sample indices
    @param min_person_freq: int, minimum number of people with outcome
    @return: boolean
    '''
    assert np.all(np.logical_or(Y == 0, Y == 1))
    num_people_with_outcome = 0
    for person_id in person_id_to_sample_idxs_dict:
        if np.sum(Y[person_id_to_sample_idxs_dict[person_id]]) > 0:
            num_people_with_outcome += 1
            if num_people_with_outcome >= min_person_freq:
                return True
    return False
    
def satisfy_auc_minimum(model,
                        X,
                        Y,
                        min_auc = .5):
    '''
    Check if AUC of model is above minimum
    @param model: model with predict_proba function, e.g. sklearn classifier
    @param X: csr matrix, sample features
    @param Y: np array, sample outcomes
    @param min_auc: float, minimum AUC
    @return: boolean
    '''
    assert X.shape[0] == len(Y)
    assert X.shape[1] == model.n_features_in_
    assert np.all(np.logical_or(Y == 0, Y == 1))
    assert min_auc    >= 0
    assert min_auc    <= 1
    
    y_pred = model.predict_proba(X)[:,1]
    auc    = roc_auc_score(Y, y_pred)
    return auc >= min_auc

def compute_auc_diff_2_models_1_dataset_patient_bootstraps(seed,
                                                           n_bootstraps,
                                                           person_ids_with_outcome,
                                                           person_ids_without_outcome,
                                                           person_id_to_sample_idxs_dict,
                                                           Y,
                                                           model_1_preds,
                                                           model_2_preds):
    '''
    Compute bootstrap estimates of the AUC difference between 2 models on 1 dataset.
    Bootstraps are at the patient-level stratified by outcome.
    @param seed: int, seed for numpy random generator
    @param n_bootstraps: int, number of bootstraps
    @param person_ids_with_outcome: list of ints, person IDs who have outcome
    @param person_ids_without_outcome: list of ints, person IDs who never have outcome
    @param person_id_to_sample_idxs_dict: dict mapping int to list of ints, person ID to sample indices
    @param Y: np array, sample outcomes
    @param model_1_preds: np array, predictions for samples from model 1
    @param model_2_preds: np array, predictions for samples from model 2
    @return: np array, bootstrap estimates of AUC differences
    '''
    np.random.seed(seed)
    bootstrap_auc_differences = np.empty(n_bootstraps)
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
        
        bootstrap_Y             = Y[bootstrap_sample_idxs]
        bootstrap_model_1_preds = model_1_preds[bootstrap_sample_idxs]
        bootstrap_model_2_preds = model_2_preds[bootstrap_sample_idxs]
        
        bootstrap_model_1_auc   = roc_auc_score(bootstrap_Y, bootstrap_model_1_preds)
        bootstrap_model_2_auc   = roc_auc_score(bootstrap_Y, bootstrap_model_2_preds)
        bootstrap_auc_diff      = bootstrap_model_1_auc - bootstrap_model_2_auc
        
        bootstrap_auc_differences[bootstrap_idx] = bootstrap_auc_diff
    
    return bootstrap_auc_differences

def satisfy_auc_diff_2_models_1_dataset_patient_bootstrap_ci_above_0(model_1,
                                                                     model_2,
                                                                     X,
                                                                     Y,
                                                                     person_id_to_sample_idxs_dict,
                                                                     logger,
                                                                     n_bootstraps = 2000):
    '''
    Check if the 90% bootstrap pivotal confidence interval of (model 1 AUC - model 2 AUC) is above 0.
    In each bootstrap, patients are drawn with replacement and all samples from that patient are included.
    Patients who have outcome at any time are drawn separately from patients who never have outcome 
    so that total number of patients who have outcome in a bootstrap iteration matches total number in population.
    @param model_1: model with predict_proba function, e.g. sklearn classifier
    @param model_2: model with predict_proba function, e.g. sklearn classifier
    @param X: csr matrix, sample features
    @param Y: np array, sample outcomes
    @param person_id_to_sample_idxs_dict: dict mapping int to list of ints, person ID to sample indices
    @param logger: logger, for INFO messages
    @param n_bootstraps: int, number of bootstraps
    @return: boolean
    '''
    start_time  = time.time()
    num_samples = len(Y)
    assert X.shape[0] == num_samples
    assert X.shape[1] == model_1.n_features_in_
    assert X.shape[1] == model_2.n_features_in_
    assert np.all(np.logical_or(Y == 0, Y == 1))
    assert n_bootstraps > 0
    logger.info(str(num_samples) + ' samples in actual dataset')
    
    # compute AUC difference on actual data
    model_1_preds   = model_1.predict_proba(X)[:,1]
    model_2_preds   = model_2.predict_proba(X)[:,1]
    model_1_auc     = roc_auc_score(Y, model_1_preds)
    model_2_auc     = roc_auc_score(Y, model_2_preds)
    actual_auc_diff = model_1_auc - model_2_auc
    logger.info('Actual AUC difference: ' + str(actual_auc_diff))
    if actual_auc_diff <= 0:
        return False
    
    # separate people who have outcome vs never have outcome
    this_start_time            = time.time()
    person_ids_with_outcome    = []
    person_ids_without_outcome = []
    for person_id in person_id_to_sample_idxs_dict:
        if np.sum(Y[person_id_to_sample_idxs_dict[person_id]]) > 0:
            person_ids_with_outcome.append(person_id)
        else:
            person_ids_without_outcome.append(person_id)
    assert len(person_ids_with_outcome)    > 0
    assert len(person_ids_without_outcome) > 0
    logger.info(str(len(person_ids_with_outcome)) + ' people have outcome')
    logger.info(str(len(person_ids_without_outcome)) + ' people never have outcome')
    logger.info('Time to separate people with and without outcome: ' + str(time.time() - this_start_time) + ' seconds')
    
    # compute AUC differences in bootstrap iterations
    this_start_time          = time.time()
    n_processes              = min(8, mp.cpu_count())
    n_bootstraps_per_process = math.ceil(n_bootstraps/float(n_processes))
    random_seeds             = np.random.randint(10000, size = n_processes)
    random_seeds             = [int(seed) for seed in random_seeds]
    with mp.get_context('spawn').Pool(processes=n_processes) as pool:
        bootstrap_auc_diffs_per_process \
            = pool.map(partial(compute_auc_diff_2_models_1_dataset_patient_bootstraps, 
                               n_bootstraps                  = n_bootstraps_per_process,
                               person_ids_with_outcome       = person_ids_with_outcome,
                               person_ids_without_outcome    = person_ids_without_outcome,
                               person_id_to_sample_idxs_dict = person_id_to_sample_idxs_dict,
                               Y                             = Y,
                               model_1_preds                 = model_1_preds,
                               model_2_preds                 = model_2_preds),
                       random_seeds)
    bootstrap_auc_differences = np.concatenate(bootstrap_auc_diffs_per_process)[:n_bootstraps]
    logger.info('Time to compute AUC differences for all bootstrap iterations: ' 
                + str(time.time() - this_start_time) + ' seconds')
        
    bootstrap_95_percentile = np.percentile(bootstrap_auc_differences, 95)
    bootstrap_ci_lb         = 2 * actual_auc_diff - bootstrap_95_percentile
    bootstrap_5_percentile  = np.percentile(bootstrap_auc_differences, 5)
    bootstrap_ci_ub         = 2 * actual_auc_diff - bootstrap_5_percentile
    
    logger.info('Bootstrap 90% confidence interval for AUC difference of 2 models on 1 dataset: ('
                + str(bootstrap_ci_lb) + ', ' + str(bootstrap_ci_ub) + ')')
    logger.info('Time to compute bootstrap confidence interval: ' + str(time.time() - start_time) + ' seconds')
    
    return bootstrap_ci_lb > 0

def compute_auc_diff_1_model_2_datasets_patient_bootstraps(seed,
                                                           n_bootstraps,
                                                           person_ids_with_outcome,
                                                           person_ids_without_outcome,
                                                           person_id_to_sample_idxs_dict_1,
                                                           person_id_to_sample_idxs_dict_2,
                                                           Y_1,
                                                           Y_2,
                                                           dataset_1_preds,
                                                           dataset_2_preds):
    '''
    Compute bootstrap estimates of the AUC difference of 1 model on 2 datasets.
    Bootstraps are at the patient level stratified by outcome.
    @param seed: int, seed for numpy random generator
    @param n_bootstraps: int, number of bootstraps
    @param person_ids_with_outcome: list of ints, person IDs who have outcome
    @param person_ids_without_outcome: list of ints, person IDs who never have outcome
    @param person_id_to_sample_idxs_dict_1: dict mapping int to list of ints, person ID to sample indices in dataset 1
    @param person_id_to_sample_idxs_dict_2: dict mapping int to list of ints, person ID to sample indices in dataset 2
    @param Y_1: np array, sample outcomes in dataset 1
    @param Y_2: np array, sample outcomes in dataset 2
    @param datatset_1_preds: np array, model predictions for samples in dataset 1
    @param dataset_2_preds: np array, model predictions for samples in dataset 2
    @return: np array, bootstrap estimates of AUC differences
    '''
    np.random.seed(seed)
    bootstrap_auc_differences = np.empty(n_bootstraps)
    for bootstrap_idx in range(n_bootstraps):
        bootstrap_person_ids_with_outcome    = np.random.choice(np.array(person_ids_with_outcome), 
                                                                len(person_ids_with_outcome))
        bootstrap_person_ids_without_outcome = np.random.choice(np.array(person_ids_without_outcome),
                                                                len(person_ids_without_outcome))
        bootstrap_person_ids                 = np.concatenate((bootstrap_person_ids_with_outcome,
                                                               bootstrap_person_ids_without_outcome))
        
        bootstrap_sample_idxs_1              = np.concatenate([np.array(person_id_to_sample_idxs_dict_1[int(person_id)],
                                                                        dtype = int)
                                                               for person_id in bootstrap_person_ids],
                                                              dtype = int)
        bootstrap_sample_idxs_2              = np.concatenate([np.array(person_id_to_sample_idxs_dict_2[int(person_id)],
                                                                        dtype = int)
                                                               for person_id in bootstrap_person_ids],
                                                              dtype = int)
        
        bootstrap_Y_1             = Y_1[bootstrap_sample_idxs_1]
        bootstrap_Y_2             = Y_2[bootstrap_sample_idxs_2]
        bootstrap_dataset_1_preds = dataset_1_preds[bootstrap_sample_idxs_1]
        bootstrap_dataset_2_preds = dataset_2_preds[bootstrap_sample_idxs_2]
        
        bootstrap_dataset_1_auc   = roc_auc_score(bootstrap_Y_1, bootstrap_dataset_1_preds)
        bootstrap_dataset_2_auc   = roc_auc_score(bootstrap_Y_2, bootstrap_dataset_2_preds)
        bootstrap_auc_diff        = bootstrap_dataset_1_auc - bootstrap_dataset_2_auc
        
        bootstrap_auc_differences[bootstrap_idx] = bootstrap_auc_diff
    return bootstrap_auc_differences

def satisfy_auc_diff_1_model_2_datasets_patient_bootstrap_ci_above_0(model,
                                                                     X_1,
                                                                     Y_1,
                                                                     person_id_to_sample_idxs_dict_1,
                                                                     X_2,
                                                                     Y_2,
                                                                     person_id_to_sample_idxs_dict_2,
                                                                     logger,
                                                                     n_bootstraps = 2000):
    '''
    Check if the 90% bootstrap pivotal confidence interval of (AUC on dataset 1 - AUC on dataset 2) is above 0.
    In each bootstrap, patients are drawn with replacement and all samples from that patient in dataset 1 
    are included in bootstrap dataset 1 and likewise for dataset 2.
    Patients who have outcome at any time are drawn separately from patients who never have outcome 
    so that total number of patients who have outcome in a bootstrap iteration matches total number in population.
    @param model: model with predict_proba function, e.g. sklearn classifier
    @param X_1: csr matrix, sample features for first set of samples
    @param Y_1: np array, sample outcomes for first set of samples
    @param person_id_to_sample_idxs_dict_1: dict mapping int to list of ints, person ID to indices in first set of samples
    @param X_2: csr matrix, sample features for second set of samples
    @param Y_2: np array, sample outcomes for second set of samples
    @param person_id_to_sample_idxs_dict_2: dict mapping int to list of ints, person ID to indices in second set of samples
    @param logger: logger, for INFO messages
    @param n_bootstraps: int, number of bootstraps
    @return: boolean
    '''
    start_time = time.time()
    num_samples_1 = len(Y_1)
    assert X_1.shape[0] == num_samples_1
    assert X_1.shape[1] == model.n_features_in_
    assert np.all(np.logical_or(Y_1 == 0, Y_1 == 1))
    num_samples_2 = len(Y_2)
    assert X_2.shape[0] == num_samples_2
    assert X_2.shape[1] == model.n_features_in_
    assert np.all(np.logical_or(Y_2 == 0, Y_2 == 1))
    assert n_bootstraps > 0
    
    # compute AUC difference on actual data
    dataset_1_preds   = model.predict_proba(X_1)[:,1]
    dataset_2_preds   = model.predict_proba(X_2)[:,1]
    dataset_1_auc     = roc_auc_score(Y_1, dataset_1_preds)
    dataset_2_auc     = roc_auc_score(Y_2, dataset_2_preds)
    actual_auc_diff   = dataset_1_auc - dataset_2_auc
    logger.info(str(num_samples_1) + ' samples in actual dataset 1')
    logger.info(str(num_samples_2) + ' samples in actual dataset 2')
    logger.info('Actual AUC difference: ' + str(actual_auc_diff))
    if actual_auc_diff <= 0:
        return False
    
    # separate people who have outcome vs never have outcome
    this_start_time            = time.time()
    all_person_ids             = set(person_id_to_sample_idxs_dict_1.keys()).union(set(person_id_to_sample_idxs_dict_2.keys()))
    person_ids_with_outcome    = []
    person_ids_without_outcome = []
    for person_id in all_person_ids:
        if np.sum(Y_1[person_id_to_sample_idxs_dict_1[person_id]]) > 0:
            person_ids_with_outcome.append(person_id)
        elif np.sum(Y_2[person_id_to_sample_idxs_dict_2[person_id]]) > 0:
            person_ids_with_outcome.append(person_id)
        else:
            person_ids_without_outcome.append(person_id)
    assert len(person_ids_with_outcome)    > 0
    assert len(person_ids_without_outcome) > 0
    logger.info(str(len(person_ids_with_outcome)) + ' people have outcome')
    logger.info(str(len(person_ids_without_outcome)) + ' people never have outcome')
    logger.info('Time to separate people with and without outcome: ' + str(time.time() - this_start_time) + ' seconds')
    
    # compute AUC differences in bootstrap iterations
    this_start_time          = time.time()
    n_processes              = min(8, mp.cpu_count())
    n_bootstraps_per_process = math.ceil(n_bootstraps/float(n_processes))
    random_seeds             = np.random.randint(10000, size = n_processes)
    random_seeds             = [int(seed) for seed in random_seeds]
    with mp.get_context('spawn').Pool(processes=n_processes) as pool:
        bootstrap_auc_diffs_per_process \
            = pool.map(partial(compute_auc_diff_1_model_2_datasets_patient_bootstraps, 
                               n_bootstraps                    = n_bootstraps_per_process,
                               person_ids_with_outcome         = person_ids_with_outcome,
                               person_ids_without_outcome      = person_ids_without_outcome,
                               person_id_to_sample_idxs_dict_1 = person_id_to_sample_idxs_dict_1,
                               person_id_to_sample_idxs_dict_2 = person_id_to_sample_idxs_dict_2,
                               Y_1                             = Y_1,
                               Y_2                             = Y_2,
                               dataset_1_preds                 = dataset_1_preds,
                               dataset_2_preds                 = dataset_2_preds),
                       random_seeds)
    bootstrap_auc_differences = np.concatenate(bootstrap_auc_diffs_per_process)[:n_bootstraps]
    logger.info('Time to compute AUC differences for all bootstrap iterations: ' 
                + str(time.time() - this_start_time) + ' seconds')
        
    bootstrap_95_percentile = np.percentile(bootstrap_auc_differences, 95)
    bootstrap_ci_lb         = 2 * actual_auc_diff - bootstrap_95_percentile
    bootstrap_5_percentile  = np.percentile(bootstrap_auc_differences, 5)
    bootstrap_ci_ub         = 2 * actual_auc_diff - bootstrap_5_percentile
    
    logger.info('Bootstrap 90% confidence interval for AUC difference of 1 model on 2 datasets: ('
                + str(bootstrap_ci_lb) + ', ' + str(bootstrap_ci_ub) + ')')
    logger.info('Time to compute bootstrap confidence interval: ' + str(time.time() - start_time) + ' seconds')
    
    return bootstrap_ci_lb > 0