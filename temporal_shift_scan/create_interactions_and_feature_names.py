import time
import numpy as np
from scipy.sparse import hstack, csr_matrix
from itertools import product

def create_interaction_terms(X,
                             Y,
                             logger,
                             get_feature_names_with_interactions = False,
                             feature_names = None,
                             outcome_name = None):
    '''
    Create interaction terms (X, Y = 0) and (X, Y = 1)
    Note that for binary features: (X = 0, Y = 0) and (X = 0, Y = 1) are not captured since there are too many entries
    Assumes only age is continuous and age is first feature
    @param X: csr matrix, # samples x # features
    @param Y: np array, # samples
    @param logger: logger, for INFO messages
    @param get_feature_names_with_interactions: bool, whether to create lists of feature names with interactions
    @param feature_names: list of str, names of features in X
    @param outcome_name: str,
    @return: 1. csr matrix containing interaction terms, # samples x (2 x # features)
             2. list of str, feature names with interactions
             3. list of str, shortened feature names with interactions
             4. list of str, feature file names with interactions
             2-4 are only returned if get_feature_names_with_interactions is True
    '''
    this_start_time   = time.time()
    assert X.shape[0] == len(Y)
    Y_csr             = csr_matrix(np.expand_dims(Y, axis=1))
    not_Y_csr         = csr_matrix(np.expand_dims(1 - Y, axis=1))
    interaction_terms = hstack((X.multiply(not_Y_csr),
                                X.multiply(Y_csr)),
                               format='csr')
    logger.info('Time to create interaction terms: ' + str(time.time() - this_start_time) + ' seconds')
    logger.info('Number of interaction terms: ' + str(interaction_terms.shape[1]))
    if not get_feature_names_with_interactions:
        return interaction_terms
    
    this_start_time                       = time.time()
    assert feature_names is not None
    assert feature_names[0] == 'age'
    assert X.shape[1] == len(feature_names)
    assert outcome_name is not None
    outcome_name_short                    = outcome_name[:min(len(outcome_name), 25)]
    outcome_file_name                     = outcome_name.replace(' ', '_')
    feature_combos                        = ['x1_y0', 
                                             'x1_y1']
    feature_names_sets                    = ['feature_names_with_interactions',
                                             'feature_names_short_with_interactions',
                                             'feature_file_names_with_interactions']
    feature_names_dict                    = {feature_combo: {feature_names_set: []
                                                             for feature_names_set in feature_names_sets}
                                             for feature_combo in feature_combos}
    outcome_names_dict                    \
        = {'y0': {'feature_names_with_interactions'      : 'no ' + outcome_name + ' (Y) ',
                  'feature_names_short_with_interactions': 'no ' + outcome_name_short + ' (Y) ',
                  'feature_file_names_with_interactions' : 'no_' + outcome_file_name  + '_Y'},
           'y1': {'feature_names_with_interactions'      : outcome_name + ' (Y) ',
                  'feature_names_short_with_interactions': outcome_name_short + ' (Y) ',
                  'feature_file_names_with_interactions' : outcome_file_name  + '_Y'}}
    for feature_name in feature_names:
        feature_name_short, feature_file_name, feature_name_full \
            = shorten_feature_name(feature_name)
        feature_name_dict = {'x1': {'feature_names_with_interactions'      : feature_name_full,
                                    'feature_names_short_with_interactions': feature_name_short,
                                    'feature_file_names_with_interactions' : feature_name_full}}
        for feature_combo, feature_names_set in product(feature_combos, feature_names_sets):
            if feature_names_set == 'feature_file_names_with_interactions':
                linker = '_and_'
            else:
                linker = ' & '
            feature_combo_name = outcome_names_dict[feature_combo[3:]][feature_names_set] + linker \
                               + feature_name_dict[feature_combo[:2]][feature_names_set]
            feature_names_dict[feature_combo][feature_names_set].append(feature_combo_name)
    
    feature_names_with_interactions       = feature_names_dict['x1_y0']['feature_names_with_interactions'] \
                                          + feature_names_dict['x1_y1']['feature_names_with_interactions']
    feature_names_short_with_interactions = feature_names_dict['x1_y0']['feature_names_short_with_interactions'] \
                                          + feature_names_dict['x1_y1']['feature_names_short_with_interactions']
    feature_file_names_with_interactions  = feature_names_dict['x1_y0']['feature_file_names_with_interactions'] \
                                          + feature_names_dict['x1_y1']['feature_file_names_with_interactions']
    logger.info('Time to create lists of feature names with interactions: ' + str(time.time() - this_start_time) + ' seconds')
    
    return interaction_terms, feature_names_with_interactions, feature_names_short_with_interactions, \
           feature_file_names_with_interactions

def shorten_feature_name(feature_name,
                         negate = False):
    '''
    Shorten feature name in 3 ways
    For gender, converts 1 to male and 0 to female
    @param feature_name: str, name of feature
    @param negate: bool, whether to create feature name for negation
    @return: 1. str, feature name up to 25 characters and number of days field
             2. str, feature name with no spaces for files
             3. str, full feature name
    '''
    race_and_ethnicity_features = {'American Indian or Alaska Native', 'Asian', 'Black or African American', 
                                   'Hispanic or Latino', 'Native Hawaiian or Other Pacific Islander', 'White'}
    feature_name_full = feature_name
    if ' - ' in feature_name:
        feature_name_fields    = feature_name.split(' - ')
        assert len(feature_name_fields) >= 4
        concept_id_field       = feature_name_fields[0]
        days_field             = feature_name_fields[-1]
        concept_name           = ' - '.join(feature_name_fields[2:-1])
        feature_name_short     = concept_name[:min(len(concept_name), 25)] + ' ' + days_field
        feature_file_name      = concept_id_field + '_' + feature_name_short.replace(' ', '_').replace('/','_')
        if negate:
            feature_name_short = 'no ' + feature_name_short
            feature_file_name  = 'no_' + feature_file_name
            feature_name_full  = 'no ' + feature_name_full
    elif feature_name == 'age':
        feature_name_short     = 'age'
        feature_file_name      = feature_name_short
    elif feature_name == 'Male':
        if negate:
            feature_name_short = 'female'
            feature_file_name  = feature_name_short
            feature_name_full  = feature_name_short
        else:
            feature_name_short = 'male'
            feature_file_name  = feature_name_short
            feature_name_full  = feature_name_short
    elif feature_name in race_and_ethnicity_features:
        if negate:
            feature_name_short = 'not ' + feature_name
        else:
            feature_name_short = feature_name
        feature_file_name      = feature_name_short.replace(' ', '_')
        feature_name_full      = feature_name_short
    else:
        assert feature_name.startswith('pred_in_')
        if negate:
            feature_name_short = 'predict not in ' + feature_name[-3:]
            feature_file_name  = 'pred_not_in_' + feature_name[-3:]
        else:
            feature_name_short = 'predict in ' + feature_name[-3:]
            feature_file_name  = feature_name
        feature_name_full  = feature_name_short
    return feature_name_short, feature_file_name, feature_name_full

def negate_interaction_name(interaction_name):
    '''
    Negate interaction term, i.e. handles adding/removing no, and to or, etc.
    @param interaction_name: str, name of original interaction
    @return: str, name corresponding to not having interaction term
    '''
    replacements = [(' (Y) & no ', ' (Y) / '),
                    ('_Y_and_no_', '_Y_or_'),
                    (' (Y) & ', ' (Y) / no '),
                    ('_Y_and_', '_Y_or_no_'),
                    (' / no predict not in ', ' / predict in '),
                    ('_or_no_predict_not_in_', '_or_predict_in_'),
                    (' / no predict in ', ' / predict not in '),
                    ('_or_no_predict_in_', '_or_predict_not_in_'),
                    (' / no age ', ' / age '),
                    ('_or_no_age_', '_or_age_'),
                    (' / no male ', ' / female '),
                    ('_or_no_male_', '_or_female_'),
                    (' / no female ', ' / male '),
                    ('_or_no_female_', '_or_male')]
    for replacement in replacements:
        interaction_name = interaction_name.replace(replacement[0], replacement[1])
    if interaction_name.startswith('no_'):
        interaction_name = interaction_name[3:]
    elif interaction_name.startswith('no '):
        interaction_name = interaction_name[3:]
    elif '_or_' in interaction_name:
        interaction_name = 'no_' + interaction_name
    else:
        interaction_name = 'no ' + interaction_name
    return interaction_name

def get_all_feature_names_without_interactions(feature_names):
    '''
    Create lists of feature names
    @param feature_names: list of str, names of features
    @return: 1. list of str, full feature names
             2. list of str, shortened feature names
             3. list of str, feature names without spaces for file names
    '''
    feature_names_full  = []
    feature_names_short = []
    feature_file_names  = []
    for feature_name in feature_names:
        feature_name_short, feature_file_name, feature_name_full \
            = shorten_feature_name(feature_name)
        feature_names_full.append(feature_name_full)
        feature_names_short.append(feature_name_short)
        feature_file_names.append(feature_file_name)
    return feature_names_full, feature_names_short, feature_file_names