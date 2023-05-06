import time
import math
import numpy as np
import multiprocessing as mp
from functools import partial
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

def compute_auc_diff_2_models_1_dataset_patient_permutation(seed,
                                                            num_permutations,
                                                            num_people,
                                                            sample_idx_to_person_idx_dict,
                                                            model_1_ranks,
                                                            model_2_ranks,
                                                            Y):
    '''
    Compute AUC difference between 2 models in each permutation.
    Permutations are swaps between two models grouped by patient.
    This method is only used for parallelization in run_auc_diff_2_models_1_dataset_patient_permutation_test.
    @param seed: int, seed for numpy random generator
    @param num_permutations: int, number of permutations to run
    @param num_people: int, number of people in dataset
    @param sample_idx_to_person_idx_dict: dict, map sample index to person index
    @param model_1_ranks: np array of float, normalized ranks of predictions from model 1
    @param model_2_ranks: np array of float, normalized ranks of predictions from model 2
    @param Y: np array of int, binary indicators for true labels 
    @return: np array of float, AUC differences in each permutation
    '''
    permutation_auc_differences = np.empty(num_permutations)
    for permutation_idx in range(num_permutations):
        swap_for_person = np.random.binomial(1, .5, size = num_people)
        swap_for_sample = np.array([swap_for_person[sample_idx_to_person_idx_dict[sample_idx]] 
                                    for sample_idx in range(len(sample_idx_to_person_idx_dict))])
        permutation_model_1_ranks = np.where(swap_for_sample == 1, model_2_ranks, model_1_ranks)
        permutation_model_2_ranks = np.where(swap_for_sample == 1, model_1_ranks, model_2_ranks)
        
        permutation_model_1_auc = roc_auc_score(Y, permutation_model_1_ranks)
        permutation_model_2_auc = roc_auc_score(Y, permutation_model_2_ranks)
        permutation_auc_diff    = permutation_model_1_auc - permutation_model_2_auc
        
        permutation_auc_differences[permutation_idx] = permutation_auc_diff
        
    return permutation_auc_differences                                     

def run_auc_diff_2_models_1_dataset_patient_permutation_test(model_1,
                                                             model_2,
                                                             X,
                                                             Y,
                                                             person_id_to_sample_idxs_dict,
                                                             logger,
                                                             num_permutations = 2000):
    '''
    Compute p-value using random permutation test (with Monte Carlo approximation)
    Null hypothesis: Model 1 has significantly higher AUC than model 2
    Assume exchangeability holds when swapping ranks of all predictions from model 1 for a patient with ranks from model 2
    @param model_1: model with predict_proba function, e.g. sklearn classifier
    @param model_2: model with predict_proba function, e.g. sklearn classifier
    @param X: csr matrix, sample features
    @param Y: np array, sample outcomes
    @param person_id_to_sample_idxs_dict: dict mapping int to list of ints, person ID to sample indices
    @param logger: logger, for INFO messages
    @param num_permutations: int, number of permutations to use in random permutation test
    @return: float, p-value
    '''
    start_time  = time.time()
    num_samples = len(Y)
    assert X.shape[0] == num_samples
    assert X.shape[1] == model_1.n_features_in_
    assert X.shape[1] == model_2.n_features_in_
    assert np.all(np.logical_or(Y == 0, Y == 1))
    assert num_permutations > 0
    
    # compute actual AUC
    this_start_time  = time.time()
    model_1_preds    = model_1.predict_proba(X)[:,1]
    model_2_preds    = model_2.predict_proba(X)[:,1]
    model_1_auc      = roc_auc_score(Y, model_1_preds)
    model_2_auc      = roc_auc_score(Y, model_2_preds)
    actual_auc_diff  = model_1_auc - model_2_auc
    logger.info('Time to compute actual AUCs: ' + str(time.time() - this_start_time) + ' seconds')
    
    # compute normalized ranks of predictions
    this_start_time  = time.time()
    model_1_ranks    = rankdata(model_1_preds)/float(num_samples)
    model_2_ranks    = rankdata(model_2_preds)/float(num_samples)
    # check ranks computed correctly
    model_1_rank_auc = roc_auc_score(Y, model_1_ranks)
    model_2_rank_auc = roc_auc_score(Y, model_2_ranks)
    assert model_1_auc == model_1_rank_auc
    assert model_2_auc == model_2_rank_auc
    logger.info('Time to compute ranks: ' + str(time.time() - this_start_time) + ' seconds')
    
    # construct sample to person index mapping
    this_start_time               = time.time()
    person_ids                    = list(person_id_to_sample_idxs_dict.keys())
    num_people                    = len(person_ids)
    sample_idx_to_person_idx_dict = dict()
    for person_idx in range(num_people):
        person_id = person_ids[person_idx]
        for sample_idx in person_id_to_sample_idxs_dict[person_id]:
            sample_idx_to_person_idx_dict[sample_idx] = person_idx
    logger.info('Time to construct sample to person index mapping: ' + str(time.time() - this_start_time) + ' seconds')
    
    # compute AUC differences in permutations
    this_start_time              = time.time()
    n_processes                  = min(8, mp.cpu_count())
    num_permutations_per_process = math.ceil(num_permutations/float(n_processes))
    random_seeds                 = np.random.randint(10000, size = n_processes)
    random_seeds                 = [int(seed) for seed in random_seeds]
    with mp.get_context('spawn').Pool(processes=n_processes) as pool:
        permutation_auc_differences_per_process \
            = pool.map(partial(compute_auc_diff_2_models_1_dataset_patient_permutation, 
                               num_permutations              = num_permutations_per_process,
                               num_people                    = num_people,
                               sample_idx_to_person_idx_dict = sample_idx_to_person_idx_dict,
                               model_1_ranks                 = model_1_ranks,
                               model_2_ranks                 = model_2_ranks,
                               Y                             = Y),
                       random_seeds)
    permutation_auc_differences = np.concatenate(permutation_auc_differences_per_process)[:num_permutations]
    logger.info('Time to compute AUC differences for all random permutations: ' 
                + str(time.time() - this_start_time) + ' seconds')
    
    num_permutations_larger = np.sum(np.where(permutation_auc_differences >= actual_auc_diff, 1, 0))
    logger.info('Time to run permutation test for 2 models on 1 dataset: ' + str(time.time() - start_time) + ' seconds')
    
    return (1 + num_permutations_larger)/float(1 + num_permutations)

def compute_auc_diff_1_model_2_datasets_patient_permutation(seed,
                                                            num_permutations,
                                                            all_person_ids,
                                                            Y_1,
                                                            dataset_1_preds,
                                                            person_id_to_sample_idxs_dict_1,
                                                            Y_2,
                                                            dataset_2_preds,
                                                            person_id_to_sample_idxs_dict_2):
    '''
    Compute AUC difference between 2 models in each permutation.
    Permutations are swaps between two datasets grouped by patient.
    This method is only used for parallelization in run_auc_diff_1_model_2_datasets_patient_permutation_test.
    @param seed: int, seed for numpy random generator
    @param num_permutations: int, number of permutations to run
    @param all_person_ids: list of int, person IDs
    @param Y_1: np array, sample outcomes in dataset 1
    @param dataset_1_preds: np array, model predictions on dataset 1
    @param person_id_to_sample_idxs_dict_1: dict mapping int to list of ints, person ID to sample indices in dataset 1
    @param Y_2: np array, sample outcomes in dataset 2
    @param dataset_2_preds: np array, model predictions on dataset 2
    @param person_id_to_sample_idxs_dict_2: dict mapping int to list of ints, person ID to sample indices in dataset 2
    @return: np array of float, AUC differences in each permutation
    '''
    permutation_auc_differences = np.empty(num_permutations)
    for permutation_idx in range(num_permutations):
        swap_for_person = np.random.binomial(1, .5, size = len(all_person_ids))
        
        permutation_dataset_1_preds = np.concatenate([dataset_2_preds[person_id_to_sample_idxs_dict_2[all_person_ids[i]]]
                                                      if swap_for_person[i] == 1
                                                      else dataset_1_preds[person_id_to_sample_idxs_dict_1[all_person_ids[i]]]
                                                      for i in range(len(all_person_ids))])
        
        permutation_Y_1             = np.concatenate([Y_2[person_id_to_sample_idxs_dict_2[all_person_ids[i]]]
                                                      if swap_for_person[i] == 1
                                                      else Y_1[person_id_to_sample_idxs_dict_1[all_person_ids[i]]]
                                                      for i in range(len(all_person_ids))])
        
        permutation_dataset_2_preds = np.concatenate([dataset_1_preds[person_id_to_sample_idxs_dict_1[all_person_ids[i]]]
                                                      if swap_for_person[i] == 1
                                                      else dataset_2_preds[person_id_to_sample_idxs_dict_2[all_person_ids[i]]]
                                                      for i in range(len(all_person_ids))])
        
        permutation_Y_2             = np.concatenate([Y_1[person_id_to_sample_idxs_dict_1[all_person_ids[i]]]
                                                      if swap_for_person[i] == 1
                                                      else Y_2[person_id_to_sample_idxs_dict_2[all_person_ids[i]]]
                                                      for i in range(len(all_person_ids))])
        
        permutation_dataset_1_auc   = roc_auc_score(permutation_Y_1, permutation_dataset_1_preds)
        permutation_dataset_2_auc   = roc_auc_score(permutation_Y_2, permutation_dataset_2_preds)
        permutation_auc_diff        = permutation_dataset_1_auc - permutation_dataset_2_auc
        
        permutation_auc_differences[permutation_idx] = permutation_auc_diff
    return permutation_auc_differences

def run_auc_diff_1_model_2_datasets_patient_permutation_test(model,
                                                             X_1,
                                                             Y_1,
                                                             person_id_to_sample_idxs_dict_1,
                                                             X_2,
                                                             Y_2,
                                                             person_id_to_sample_idxs_dict_2,
                                                             logger,
                                                             num_permutations = 2000):
    '''
    Compute p-value using random permutation test (with Monte Carlo approximation)
    Null hypothesis: Model has significantly higher AUC on dataset 1 than dataset 2
    Assume exchangeability holds when all samples for a patient in dataset 1 are swapped with samples in dataset 2
    @param model: model with predict_proba function, e.g. sklearn classifier
    @param X_1: csr matrix, sample features in dataset 1
    @param Y_1: np array, sample outcomes in dataset 1
    @param person_id_to_sample_idxs_dict_1: dict mapping int to list of ints, person ID to sample indices in dataset 1
    @param X_2: csr matrix, sample features in dataset 2
    @param Y_2: np array, sample outcomes in dataset 2
    @param person_id_to_sample_idxs_dict_2: dict mapping int to list of ints, person ID to sample indices in dataset 2
    @param logger: logger, for INFO messages
    @param num_permutations: int, number of permutations to use in random permutation test
    @return: float, p-value
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
    assert num_permutations > 0
    
    # compute AUC difference on actual data
    this_start_time   = time.time()
    dataset_1_preds   = model.predict_proba(X_1)[:,1]
    dataset_2_preds   = model.predict_proba(X_2)[:,1]
    dataset_1_auc     = roc_auc_score(Y_1, dataset_1_preds)
    dataset_2_auc     = roc_auc_score(Y_2, dataset_2_preds)
    actual_auc_diff   = dataset_1_auc - dataset_2_auc
    all_person_ids    = list(set(person_id_to_sample_idxs_dict_1.keys()).union(set(person_id_to_sample_idxs_dict_2.keys())))
    logger.info('Time to compute AUC difference on actual data: ' + str(time.time() - this_start_time) + ' seconds')
    
    # compute AUC differences in permutations
    this_start_time              = time.time()
    n_processes                  = min(8, mp.cpu_count())
    num_permutations_per_process = math.ceil(num_permutations/float(n_processes))
    random_seeds                 = np.random.randint(10000, size = n_processes)
    random_seeds                 = [int(seed) for seed in random_seeds]
    with mp.get_context('spawn').Pool(processes=n_processes) as pool:
        permutation_auc_differences_per_process \
            = pool.map(partial(compute_auc_diff_1_model_2_datasets_patient_permutation, 
                               num_permutations                = num_permutations_per_process,
                               all_person_ids                  = all_person_ids,
                               Y_1                             = Y_1,
                               dataset_1_preds                 = dataset_1_preds,
                               person_id_to_sample_idxs_dict_1 = person_id_to_sample_idxs_dict_1,
                               Y_2                             = Y_2,
                               dataset_2_preds                 = dataset_2_preds,
                               person_id_to_sample_idxs_dict_2 = person_id_to_sample_idxs_dict_2),
                           random_seeds)
        
    permutation_auc_differences = np.concatenate(permutation_auc_differences_per_process)[:num_permutations]
    logger.info('Time to compute AUC differences for all random permutations: ' 
                + str(time.time() - this_start_time) + ' seconds')
    
    num_permutations_larger = np.sum(np.where(permutation_auc_differences >= actual_auc_diff, 1, 0))
    logger.info('Time to run permutation test for 1 model on 2 datasets: ' + str(time.time() - start_time) + ' seconds')
    
    return (1 + num_permutations_larger)/float(1 + num_permutations)
    
def run_benjamini_hochberg(p_value_df,
                           fdr = .05):
    '''
    Run Benjamini-Hochberg multiple hypothesis test 
    to identify significant hypotheses that satisfy false discovery rate
    @param p_value_df: pandas dataframe, contains column called 'P-values' with float p-values from hypothesis tests,
                       will be modified and returned
    @param fdr: float, desired false discovery rate
    @return: pandas dataframe indicating which hypotheses were accepted
    '''
    p_values = p_value_df['P-value'].values
    assert np.all(np.logical_and(p_values >= 0, p_values <= 1))
    assert fdr >= 0
    assert fdr <= 1
    
    p_value_ranks                  = rankdata(p_values, method = 'min')
    critical_vals                  = p_value_ranks / float(len(p_values)) * fdr
    p_value_df['Rank']             = p_value_ranks
    p_value_df['Critical value']   = critical_vals
    p_value_df.sort_values(by      = 'Rank',
                           inplace = True)
    accept_to_rank                 = p_value_df.loc[p_value_df['P-value'] < p_value_df['Critical value']]['Rank'].max()
    p_value_df['Accept']           = np.where(p_value_df['Rank'] <= accept_to_rank, 1, 0)
    return p_value_df