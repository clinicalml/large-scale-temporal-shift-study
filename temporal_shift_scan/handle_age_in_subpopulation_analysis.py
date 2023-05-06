import os
import math
import time
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

def get_region_age_bound(train_valid_age,
                         coefficient,
                         dataset_file_header,
                         age_in_original_scale,
                         logger):
    '''
    Get boundary of region defined by age
    If coefficient is negative, region is below 20th percentile
    If coefficient is positive, region is above 80th percentile
    Adjust boundary to integer age in original scale if available
    Age scaling parameters in file dataset_file_header + _age_scaling_params.txt: mean, std
    If age scaling parameters are saved, compute boundary in original and normalized scales
    @param train_valid_age: np array of age after scaling in training and validation sets
    @param coefficient: float, coefficient in region logistic regression
    @param dataset_file_header: str, start of path where age scaling parameters are saved
    @param age_in_original_scale: bool, whether age is in original scale in train_valid_age (i.e. not normalized)
    @param logger: logger, for INFO messages
    @return: 1. float, boundary after scaling
             2. float, boundary in original age scale
    '''
    this_start_time = time.time()
    assert coefficient != 0
    if coefficient > 0:
        region_age_bound   = np.percentile(train_valid_age, 80)
    else:
        region_age_bound   = np.percentile(train_valid_age, 20)
        
    # set boundary for scale of train_valid_age
    if age_in_original_scale:
        region_age_bound_original     = region_age_bound
        if coefficient > 0:
            region_age_bound_original = math.ceil(region_age_bound_original)
        else:
            region_age_bound_original = math.floor(region_age_bound_original)
    else:
        region_age_bound_scaled       = region_age_bound
    
    # adjust boundary to other scale
    age_scaler_file_name   = dataset_file_header + '_age_scaling_params.txt'
    if os.path.exists(age_scaler_file_name):
        logger.info('Parameters for re-scaling age available at ' + age_scaler_file_name)
        with open(age_scaler_file_name, 'r') as f:
            age_params     = f.read().strip().split(',')
        age_mean           = float(age_params[0])
        age_std            = float(age_params[1])
        
        if age_in_original_scale:
            region_age_bound_scaled       = (region_age_bound_original - age_mean) / age_std
        else:
            region_age_bound_original     = age_mean + age_std * region_age_bound_scaled
            if coefficient > 0:
                region_age_bound_original = math.ceil(region_age_bound_original)
            else:
                region_age_bound_original = math.floor(region_age_bound_original)
            region_age_bound_scaled       = (region_age_bound_original - age_mean) / age_std
    else:
        logger.info('Parameters for re-scaling age are not available at ' + age_scaler_file_name)
        if age_in_original_scale:
            region_age_bound_original     = region_age_bound_scaled
        else:
            region_age_bound_scaled       = region_age_bound_original
    logger.info('Time to get region age boundary: ' + str(time.time() - this_start_time) + ' seconds')
    return region_age_bound_scaled, region_age_bound_original