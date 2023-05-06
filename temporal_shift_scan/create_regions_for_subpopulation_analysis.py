import numpy as np
import os
import sys
from copy import deepcopy
from scipy.sparse import hstack, csc_matrix
from create_interactions_and_feature_names import (
    negate_interaction_name,
    create_interaction_terms
)
from os.path import dirname, abspath, join
sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from h5py_utils import load_sparse_matrix_from_h5py, save_sparse_matrix_to_h5py

def construct_region_name(feature_names,
                          coefficient,
                          age_region_boundary = 0):
    '''
    Create a dictionary containing 
    @param feature_names: dict mapping str to str, keys include feature_name_full, feature_name_short, feature_file_name
    @param coefficient: float, positive or negative to indicate direction
    @param age_region_boundary: int or float, boundary of age for defining region, 0 if age is not feature
    @return: dict mapping str to str, 3 names for region and not region
    '''
    assert {'feature_name_full', 'feature_name_short', 'feature_file_name'}.issubset(set(feature_names.keys()))
    assert coefficient != 0
    feature_names = deepcopy(feature_names)
    if ' (Y) & ' in feature_names['feature_name_short']:
        negated_feature_name_full      = negate_interaction_name(feature_names['feature_name_full'])
        negated_feature_name_short     = negate_interaction_name(feature_names['feature_name_short'])
        negated_feature_file_name      = negate_interaction_name(feature_names['feature_file_name'])
    else:
        if feature_names['feature_name_short'] == 'male':
            negated_feature_name_full  = 'female'
            negated_feature_name_short = 'female'
            negated_feature_file_name  = 'female'
        elif feature_names['feature_file_name'].startswith('pred_in_'):
            negated_feature_name_full  = 'predicted not in ' + feature_names['feature_file_name'][-3:]
            negated_feature_name_short = negated_feature_name_full
            negated_feature_file_name  = 'pred_not_in_' + feature_names['feature_file_name'][-3:]
        elif feature_names['feature_name_short'] == 'age':
            negated_feature_name_full  = 'age'
            negated_feature_name_short = 'age'
            negated_feature_file_name  = 'age'
        else:
            negated_feature_name_full  = 'no ' + feature_names['feature_name_full']
            negated_feature_name_short = 'no ' + feature_names['feature_name_short']
            negated_feature_file_name  = 'no_' + feature_names['feature_file_name']
    
    coefficient_handled     = False
    if feature_names['feature_name_short'].endswith('age'):
        coefficient_handled = True
        if coefficient > 0:
            feature_names['feature_name_full']  += ' >= ' + str(age_region_boundary)
            feature_names['feature_name_short'] += ' >= ' + str(age_region_boundary)
            feature_names['feature_file_name']  += '_atleast_' + str(age_region_boundary)
            negated_feature_name_full           += ' < '  + str(age_region_boundary)
            negated_feature_name_short          += ' < '  + str(age_region_boundary)
            negated_feature_file_name           += '_below_' + str(age_region_boundary)
        else:
            feature_names['feature_name_full']  += ' <= ' + str(age_region_boundary)
            feature_names['feature_name_short'] += ' <= ' + str(age_region_boundary)
            feature_names['feature_file_name']  += '_atmost_' + str(age_region_boundary)
            negated_feature_name_full           += ' > '  + str(age_region_boundary)
            negated_feature_name_short          += ' > '  + str(age_region_boundary)
            negated_feature_file_name           += '_above_' + str(age_region_boundary)
    if coefficient > 0 or coefficient_handled:
        region_names = {'region_name_full'     : feature_names['feature_name_full'],
                        'region_name_short'    : feature_names['feature_name_short'],
                        'region_file_name'     : feature_names['feature_file_name'],
                        'not_region_name_full' : negated_feature_name_full,
                        'not_region_name_short': negated_feature_name_short,
                        'not_region_file_name' : negated_feature_file_name}
    else:
        region_names = {'region_name_full'     : negated_feature_name_full,
                        'region_name_short'    : negated_feature_name_short,
                        'region_file_name'     : negated_feature_file_name,
                        'not_region_name_full' : feature_names['feature_name_full'],
                        'not_region_name_short': feature_names['feature_name_short'],
                        'not_region_file_name' : feature_names['feature_file_name']}
    return region_names

def get_region_X(X,
                 Y,
                 get_interaction_terms,
                 interaction_terms_saved_path,
                 logger):
    '''
    Create interaction terms and concatenate X
    Load from file if exists. Else save to file
    @param X: csr matrix, # samples x # features
    @param Y: np array, # samples
    @param interaction_terms_saved_path: str, path to h5py file where interaction terms are saved or loaded from if exists
    @param get_interaction_terms: bool, whether to get interaction terms
    @param logger: logger, for INFO messages
    @return: csr matrix, # samples x # new features
    '''
    assert X.shape[0] == len(Y)
    if get_interaction_terms:
        if os.path.exists(interaction_terms_saved_path):
            logger.info('Interactions loaded from ' + interaction_terms_saved_path)
            region_X = load_sparse_matrix_from_h5py(interaction_terms_saved_path)
        else:
            region_X = hstack((create_interaction_terms(X,
                                                       Y,
                                                       logger),
                               np.expand_dims(Y, axis=1)),
                              format = 'csr')
            save_sparse_matrix_to_h5py(interaction_terms_saved_path,
                                       region_X)
            logger.info('Interactions not found. Saved to ' + interaction_terms_saved_path)
    else:
        region_X = hstack((X,
                           np.expand_dims(Y, axis=1)),
                          format = 'csr')
    return region_X

def create_single_feature_region_indicator_function(feature_idx, 
                                                    coefficient, 
                                                    get_interaction_terms,
                                                    logger,
                                                    age_region_boundary=0):
    '''
    Create function that takes in X and Y and outputs binary indicators for whether each sample is in the region
    Region is defined as having or not having a single feature
    @param feature_idx: int, index of feature in X after creating interaction terms that is defining region
    @param coefficient: float, positive or negative to indicate direction
    @param get_interaction_terms: bool, whether to get interaction terms
    @param logger: logger, for INFO messages
    @param age_region_boundary: int or float, boundary of age for defining region, 0 if age is not feature
    @return: function
    '''
    assert feature_idx >= 0
    assert coefficient != 0
    feature_is_age = False
    if feature_idx == 0 or (get_interaction_terms and feature_idx == 1):
        feature_is_age = True
        assert age_region_boundary > 0
    def get_region_indicators(X,
                              Y,
                              interaction_terms_saved_path):
        '''
        Get region indicators as defined by feature and outcome
        @param X: csr matrix, # samples x # features
        @param Y: np array, # samples
        @param interaction_terms_saved_path: str, path to h5py file where interaction terms are saved or loaded from if exists
        @return: np array of binary indicators
        '''
        region_X = get_region_X(X,
                                Y,
                                get_interaction_terms,
                                interaction_terms_saved_path,
                                logger)
        assert region_X.shape[1] > feature_idx
        feature_X = csc_matrix(region_X)[:,feature_idx].toarray()
        if feature_is_age:
            if coefficient > 0:
                return np.where(feature_X >= age_region_boundary, 1, 0)
            return np.where(feature_X <= age_region_boundary, 1, 0)
        if coefficient > 0:
            return feature_X
        return np.where(feature_X == 0, 1, 0)
    return get_region_indicators

def create_model_based_region_indicator_function(model,
                                                 threshold,
                                                 get_interaction_terms,
                                                 logger,
                                                 get_probabilities = False):
    '''
    Create function that takes in X and Y and outputs binary indicators for whether each sample is in the region
    Region is determined by model prediction
    If get probabilities, function outputs probabilities for whether each sample is in the region
    @param model: any model with predict_proba function that outputs probabilities for whether sample is in region,
                  e.g. sklearn LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
    @param threshold: float, value for thresholding probabilities predicted by decision tree
    @param get_interaction_terms: bool, whether dectree was trained with interaction terms
    @param logger: logger, for INFO messages
    @param get_probabilities: bool, whether function outputs probabilities or indicators for whether samples are in region
    @return: function
    '''
    def get_region_indicators(X,
                              Y,
                              interaction_terms_saved_path):
        '''
        Get region indicators as defined by feature and outcome
        @param X: csr matrix, # samples x # features
        @param Y: np array, # samples
        @param interaction_terms_saved_path: str, path to h5py file where interaction terms are saved or loaded from if exists
        @return: np array of binary indicators or probabilities
        '''
        region_X     = get_region_X(X,
                                    Y,
                                    get_interaction_terms,
                                    interaction_terms_saved_path,
                                    logger)
        assert region_X.shape[1] == model.n_features_in_
        region_preds = model.predict_proba(region_X)[:,1]
        if get_probabilities:
            return region_preds
        return np.where(region_preds >= threshold, 1, 0)
    return get_region_indicators