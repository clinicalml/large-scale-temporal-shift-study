import json
import sys
from datetime import datetime
from os.path import dirname, abspath, join

sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from logging_utils import set_up_logger
from h5py_utils import load_data_from_h5py

def compute_number_unique_labs(outcome_specific_data_dir,
                               logger):
    '''
    Compute number of lab concepts in covariate set
    @param outcome_specific_data_dir: str, path to outcome data directory
    @param logger: logger, for INFO messages
    @return: None
    '''
    # load lab feature names
    outcome_full_dir = config.outcome_data_dir + outcome_specific_data_dir
    if outcome_specific_data_dir.startswith('dataset_condition'):
        freq_str     = 'freq100'
    else:
        freq_str     = 'freq300'
    with open(outcome_full_dir + 'fold0_' + freq_str + '_labs_feature_names.json', 'r') as f:
        lab_feature_names = json.load(f)

    # compute number of lab types
    lab_concept_ids = set()
    for lab_name in lab_feature_names:
        lab_concept_ids.add(lab_name.split(' - lab -')[0])
    logger.info(str(len(lab_concept_ids)) + ' lab types in ' + outcome_specific_data_dir + ' outcome set-up')
    
def compute_number_features(outcome_specific_data_dir,
                            logger):
    '''
    Compute number of covariates
    @param outcome_specific_data_dir: str, path to outcome data directory
    @param logger: logger, for INFO messages
    @return: None
    '''
    # compute number of features from name lists
    outcome_full_dir   = config.outcome_data_dir + outcome_specific_data_dir
    feature_types      = ['conditions', 'procedures', 'labs', 'drugs', 'specialties', 'general', 'age']
    total_num_features = 0
    if outcome_specific_data_dir.startswith('dataset_condition'):
        freq_str       = 'freq100'
    else:
        freq_str       = 'freq300'
    for feature_type in feature_types:
        with open(outcome_full_dir + 'fold0_' + freq_str + '_' + feature_type + '_feature_names.json', 'r') as f:
            total_num_features += len(json.load(f))
    logger.info(str(total_num_features) + ' features in ' + outcome_specific_data_dir + ' outcome set-up')
    
def compute_patient_statistics(eligibility_matrix_filename,
                               logger):
    '''
    Compute number of patients and number of samples per patient in set-up
    @param eligibility_matrix_filename: str, hf5 file containing eligibility matrix
    @param logger: logger, for INFO messages
    @return: None
    '''
    # load eligibility matrix
    eligibility_matrix = load_data_from_h5py(eligibility_matrix_filename)['monthly_eligibility']
    
    # number of patients
    logger.info(str(eligibility_matrix.shape[0]) + ' patients in ' + eligibility_matrix_filename)
    
    # number of samples per patient
    logger.info(str(float(eligibility_matrix.sum())/eligibility_matrix.shape[0]) + ' samples per patient in '
                + eligibility_matrix_filename)

if __name__ == '__main__':
    
    logging_filename = config.logging_dir + 'cohort_statistics_' \
                     + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.log'
    logger           = set_up_logger('logger_main',
                                     logging_filename)
    
    top_lab_outcome_name       = 'dataset_lab_3009744_low_outcomes/'
    top_condition_outcome_name = 'dataset_condition_254761_outcomes/'
    
    compute_number_unique_labs(top_lab_outcome_name,
                               logger)
    compute_number_features(top_lab_outcome_name,
                            logger)
    compute_number_features(top_condition_outcome_name,
                            logger)
    
    lab_eligibility_matrix_name                 = config.omop_data_dir + 'eligibility_1_year.hf5'
    condition_eligibility_matrix_name           = config.omop_data_dir + 'eligibility_3_years.hf5'
    condition_reindexed_eligibility_matrix_name = config.outcome_data_dir + top_condition_outcome_name \
                                                + 'condition_254761_eligibility.hf5'
    
    compute_patient_statistics(lab_eligibility_matrix_name,
                               logger)
    compute_patient_statistics(condition_eligibility_matrix_name,
                               logger)
    compute_patient_statistics(condition_reindexed_eligibility_matrix_name,
                               logger)