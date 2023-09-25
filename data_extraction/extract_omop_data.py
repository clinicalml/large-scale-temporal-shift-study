import pickle
import sys
import numpy as np
import pandas as pd
import os
from pathlib import Path
from datetime import datetime, timedelta
from scipy.sparse import hstack, vstack, csr_matrix, csc_matrix
from sklearn.model_selection import train_test_split, KFold
import time
from sklearn.preprocessing import StandardScaler
import gc
from itertools import product
import time
import h5py
import json
import matplotlib.pyplot as plt
import sqlalchemy
from copy import deepcopy
from os.path import dirname, abspath, join
from collections import defaultdict

sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from h5py_utils import *
from db_utils import session_scope, check_date_format
from omop_windowed_utils import build_3d_sparse_feature_matrix

from omop_learn.omop import OMOPDataset
from omop_learn.data.cohort import Cohort
from omop_learn.data.feature import Feature
from omop_learn.data.common import ConceptTokenizer
from omop_learn.utils.dbutils import Database
from omop_learn.utils.data_utils import to_unixtime, from_unixtime
from omop_learn.utils.config import Config as OMOPConfig
        
def create_window_prediction_dates(eligibility_time):
    '''
    Database changes:
    0. Expects {nonstationarity_schema_name}.prediction_dates table with 'prediction_date' column
    1. Creates {nonstationarity_schema_name}.prediction_dates_{window} with 'prediction_date' column and only dates in range
    {window} is eligibility time with '_' replacing ' '
    If eligibility time is 1 year, range is 2015-01-01 to 2020-12-31
    If eligibility time is 3 year, range is 2017-01-01 to 2020-12-31
    Other eligibility times are not supported.
    @param eligibility_time: str, 1 year or 3 years
    @return: None
    '''
    assert eligibility_time in {'1 year', '3 years'}
    prediction_dates_window_sql = ('DROP TABLE IF EXISTS {schema_name}.prediction_dates_{window};'
                                   'CREATE TABLE {schema_name}.prediction_dates_{window} AS '
                                   'SELECT * '
                                   'FROM {schema_name}.prediction_dates '
                                   'WHERE DATE(prediction_date) >= DATE(\'{start_date}\') '
                                   'AND DATE(prediction_date) <= DATE(\'{end_date}\');'
                                  )
    sql_params = {'end_date'   : '2020-12-31',
                  'schema_name': config.nonstationarity_schema_name}
    if eligibility_time == '1 year':
        sql_params.update({'window'     : '1_year',
                           'start_date' : '2015-01-01'})
    else:
        sql_params.update({'window'    : '3_years',
                           'start_date': '2017-01-01'})
    prediction_dates_window_sql = prediction_dates_window_sql.format(**sql_params)
    engine = config.create_sqlalchemy_engine()
    with session_scope(engine) as session:
        session.execute(sqlalchemy.text(prediction_dates_window_sql))
        session.commit()
        
def _get_prediction_dates(eligibility_time):
    '''
    Get prediction dates in {nonstationarity_schema_name}.prediction_dates_{window}
    @param eligibility_time: str, require observed for at least 95% of this time prior to prediction date, 
                             e.g. 1 year or 3 years, {window} is same time with '_' replacing ' '
    @return: list of str of prediction dates
    '''
    prediction_dates_sql = ('SELECT prediction_date '
                            'FROM {schema_name}.prediction_dates_{window} '
                            'ORDER BY prediction_date;'
                           )
    prediction_dates_sql = prediction_dates_sql.format(window      = eligibility_time.replace(' ', '_'),
                                                       schema_name = config.nonstationarity_schema_name)
    engine = config.create_sqlalchemy_engine()
    with session_scope(engine) as session:
        prediction_dates_result = session.execute(sqlalchemy.text(prediction_dates_sql))
        session.commit()
        prediction_dates        = [str(row['prediction_date']) for row in prediction_dates_result]
    return prediction_dates
    
def extract_monthly_eligibility(monthly_eligibility_hf5_file,
                                eligible_ids_hf5_file,
                                eligibility_time,
                                eol_version,
                                logger):
    '''
    Extract cohort of patients eligible at each prediction date
    Database changes:
    0. Expects {nonstationarity_schema_name}.prediction_dates_{window} table with 'prediction_date' column
    1. Creates {nonstationarity_schema_name}.monthly_eligiblity_{window} table in omop pkg cohort format. 
       One row per eligible person-prediction date.
    @param monthly_eligibility_hf5_file: str, path to file where monthly_eligibility will be read or stored
    @param eligible_ids_hf5_file: str, path to file where eligible_ids will be read or stored
    @param eligibility_time: str, require observed for at least 95% of this time prior to prediction date, 
                             e.g. 1 year or 3 years, {window} is same time with '_' replacing ' '
    @param eol_version: bool, will add Medicare Advantage and age > 70 eligibility criteria if True
    @param logger: logger, for INFO messages
    @return: 1. monthly_eligibility: numpy array of patient by time with binary indicators for whether patient is eligible 
                                     at that time
             2. eligible_ids: List of ids of patients who are eligible at any time point, order of rows in 1
    '''
    if os.path.exists(monthly_eligibility_hf5_file) and os.path.exists(eligible_ids_hf5_file):
        logger.info('Loading monthly eligibility from ' + monthly_eligibility_hf5_file + ', ' + eligible_ids_hf5_file)
        monthly_eligibility = load_data_from_h5py(monthly_eligibility_hf5_file)['monthly_eligibility']
        eligible_ids        = load_data_from_h5py(eligible_ids_hf5_file)['eligible_ids']
        return monthly_eligibility, eligible_ids
    
    engine = config.create_sqlalchemy_engine()
    with session_scope(engine) as session:
        # create table of person_id, prediction_date, and other cohort columns where person is eligible on prediction date
        if eol_version:
            sql_file_path = 'sql/select_patient_eligibility_eol.sql'
            eol_suffix    = '_eol'
        else:
            sql_file_path = 'sql/select_patient_eligibility.sql'
            eol_suffix    = ''
        with open(join(dirname(abspath(__file__)), sql_file_path), 'r') as f:
            create_eligibility_table_sql = f.read()
        create_eligibility_table_sql     \
            = create_eligibility_table_sql.format(window_name       = eligibility_time.replace(' ', '_'),
                                                  window            = eligibility_time,
                                                  schema_name       = config.nonstationarity_schema_name)
        session.execute(sqlalchemy.text(create_eligibility_table_sql))
        session.commit()
            
        # map prediction date to column index of array
        prediction_dates = {date: i for i, date in enumerate(_get_prediction_dates(eligibility_time))}

        # select # of patients to determine size of monthly eligibility array
        num_patients_sql = ('SELECT COUNT(DISTINCT person_id) '
                            'FROM {schema_name}.monthly_eligibility_{window_name}{eol_suffix};'
                           )
        num_patients_sql = num_patients_sql.format(window_name  = eligibility_time.replace(' ', '_'),
                                                   eol_suffix   = eol_suffix,
                                                   schema_name  = config.nonstationarity_schema_name)
        num_patients_result = session.execute(sqlalchemy.text(num_patients_sql)).fetchone()
        session.commit()
        num_patients = int(num_patients_result['count'])

        # create array of patients x prediction dates
        # binary indicators for whether patient is eligible that month
        monthly_eligibility = np.zeros((num_patients, len(prediction_dates)), dtype=np.int8)
        select_eligible_sql = ('SELECT person_id, '
                                      'end_date '
                               'FROM {schema_name}.monthly_eligibility_{window_name}{eol_suffix} '
                               'ORDER BY person_id, '
                                        'end_date;'
                              )
        select_eligible_sql = select_eligible_sql.format(window_name  = eligibility_time.replace(' ', '_'),
                                                         eol_suffix   = eol_suffix,
                                                         schema_name  = config.nonstationarity_schema_name)
        select_eligible_result = session.execute(sqlalchemy.text(select_eligible_sql))
        session.commit()
        eligible_person_idx = -1
        prev_eligible_person_id = -100
        eligible_ids = []
        for row in select_eligible_result:
            if row['person_id'] != prev_eligible_person_id:
                eligible_person_idx += 1
                prev_eligible_person_id = row['person_id']
                eligible_ids.append(row['person_id'])
            prediction_date_idx = prediction_dates[row['end_date'].strftime('%Y-%m-%d')]
            monthly_eligibility[eligible_person_idx, prediction_date_idx] = 1
        assert eligible_person_idx == monthly_eligibility.shape[0] - 1
        
    save_data_to_h5py(monthly_eligibility_hf5_file,
                      {'monthly_eligibility': monthly_eligibility})
    save_data_to_h5py(eligible_ids_hf5_file,
                      {'eligible_ids': eligible_ids})
    return monthly_eligibility, eligible_ids
    
def extract_omop_cohort(cohort_pickle_file,
                        eligibility_time,
                        eol_version,
                        logger,
                        debug_size = None):
    '''
    Extract cohort of patients with final prediction date to use for gathering all features
    Database changes:
    0. Expects {schema_name}.prediction_dates_{window} table with 'prediction_date' column
    1. Expects {schema_name}.monthly_eligiblity_{window} table in omop pkg cohort format. 
       One row per eligible person-prediction date.
    2. Creates {schema_name}.omop_cohort_{window} table in omop pkg cohort format.
       One row per person in {schema_name}.monthly_eligiblity_{window} and final prediction date.
    @param cohort_pickle_file: str, path to file where cohort object is stored, will read this file if available, else write it
    @param eligibility_time: str, require observed for at least 95% of this time prior to prediction date, 
                             e.g. 1 year or 3 years, {window} is same time with '_' replacing ' '
    @param eol_version: bool, uses patient eligibility table with additional criteria for eol if True
    @param logger: logger, for INFO messages
    @param debug_size: int, specifies limited cohort size for debugging, otherwise None
    @return: omop pkg cohort
    '''
    if os.path.exists(cohort_pickle_file):
        logger.info('Loading cohort from ' + cohort_pickle_file)
        with open(cohort_pickle_file, 'rb') as f:
            cohort = pickle.load(f)
        return cohort
    
    if eol_version:
        eol_suffix = '_eol'
    else:
        eol_suffix = ''
    if debug_size is not None:
        debug_suffix   = '_debug' + str(debug_size)
        debug_size_str = str(debug_size)
    else:
        debug_suffix   = ''
        debug_size_str = ''
    
    engine = config.create_sqlalchemy_engine()
    with session_scope(engine) as session:
        
        # create omop pkg cohort
        omop_config  = OMOPConfig({"path"           : "postgresql://" + config.db_name,
                                   "cdm_schema"     : "cdm",
                                   "aux_cdm_schema" : "cdm_aux",
                                   "prefix_schema"  : config.nonstationarity_schema_name})
        connect_args = {"host": '/var/run/postgresql/'}
        db           = Database(omop_config,
                                connect_args)
        window_name  = eligibility_time.replace(' ', '_')
        
        cohort_sql_params = {'cohort_table_name': 'omop_cohort_' + window_name + eol_suffix,
                             'schema_name'      : config.nonstationarity_schema_name,
                             'window_name'      : window_name,
                             'eol_suffix'       : eol_suffix,
                             'debug_suffix'     : debug_suffix,
                             'debug_size'       : debug_size_str}
        
        check_if_cohort_table_exists_sql = ('SELECT EXISTS ( '
                                            'SELECT FROM pg_tables '
                                            'WHERE schemaname = \'{schema_name}\' '
                                            'AND tablename = \'{schema_name}.omop_cohort_{window_name}{eol_suffix}\' '
                                            ');'
                                           )
        check_if_cohort_table_exists_sql = check_if_cohort_table_exists_sql.format(**cohort_sql_params)
        cohort_table_exists      = session.execute(check_if_cohort_table_exists_sql)
        if cohort_table_exists:
            cohort_sql_file_name = 'sql/select_omop_cohort_from_existing_table.sql'
        else:
            cohort_sql_file_name = 'sql/select_omop_cohort.sql'
        with open(join(dirname(abspath(__file__)), cohort_sql_file_name), 'r') as f:
            cohort   = Cohort.from_sql_file(f, 
                                            config, 
                                            db, 
                                            params=cohort_sql_params)
            
        if debug_size is not None:
            with open(join(dirname(absphat(__file__)), 'sql/select_omop_cohort_debug.sql'), 'r') as f:
                cohort   = Cohort.from_sql_file(f, 
                                                config, 
                                                db, 
                                                params=cohort_sql_params)
        session.commit()
        
    with open(cohort_pickle_file, 'wb') as f:
        pickle.dump(cohort, 
                    f, 
                    protocol=3)
    return cohort

def extract_features(cohort,
                     features_hf5_file,
                     features_auxiliary_hf5_file,
                     dataset_json_dir,
                     eligibility_time,
                     eol_version,
                     logger,
                     debug_size = None):
    ''''
    Extract feature matrix for cohort up to final time point
    Expects table {nonstationarity_schema_name}.omop_cohort_{window} in database
    Will read from pickle or json where available, else write to them
    For now, not writing to pickle file because matrix size is too large
    @param cohort: Cohort object
    @param features_hf5_file: str, path to file where feature matrix and time list are stored,
                              will read this file if available
    @param features_auxiliary_hf5_file: str, path to file to store time and person ID lists
    @param dataset_json_dir: str, name of directory where omop dataset json is stored
    @param eligibility_time: str, require observed for at least 95% of this time prior to prediction date, 
                             e.g. 1 year or 3 years, {window} is same time with '_' replacing ' '
    @param eol_version: bool, uses patient eligibility table with additional criteria for eol if True
    @param logger: logger, for INFO messages
    @param debug_size: int, specifies limited cohort size for debugging, otherwise None
    @return: 1. feature_matrix: 3d sparse tensor
             2. time_list: list of dates for dim 1 in feature_matrix
             3. person_ids: list of person IDs for dim 0 in feature_matrix
    '''
    if os.path.exists(features_hf5_file) and os.path.exists(features_auxiliary_hf5_file):
        logger.info('Loading features from ' + features_hf5_file + ', ' + features_auxiliary_hf5_file)
        feature_matrix = load_coo_matrix_from_h5py(features_hf5_file)
        auxiliary_data = load_data_from_h5py(features_auxiliary_hf5_file)
        times_list = auxiliary_data['times_list']
        person_ids = auxiliary_data['person_ids']
        return feature_matrix, times_list, person_ids
    
    if dataset_json_dir[-1] != '/':
        dataset_json_dir += '/'
    cache_path = Path(config.cache_dir)
    
    np.random.seed(1807)
    # specify temporal features to extract - specify labs first because has with clause
    feature_paths  = ["sql/labs.sql",
                      "sql/drugs.sql",
                      "sql/conditions.sql",
                      "sql/procedures.sql",
                      "sql/specialties.sql"
                     ]
    feature_paths  = [join(dirname(abspath(__file__)), path)
                      for path in feature_paths]
    feature_names  = ["labs",
                      "drugs", 
                      "conditions", 
                      "procedures", 
                      "specialties" 
                     ]
    eol_suffix     = ''
    if eol_version:
        eol_suffix = '_eol'
    debug_suffix     = ''
    if debug_size is not None:
        debug_suffix = '_debug' + str(debug_size)
    feature_sql_params = {'window_name'           : eligibility_time.replace(' ', '_'),
                          'eol_suffix'            : eol_suffix,
                          'schema_name'           : config.nonstationarity_schema_name,
                          'measurement_aux_schema': config.measurement_aux_schema,
                          'debug_suffix'          : debug_suffix}
    features = [Feature(n, p, params=feature_sql_params) 
                for n, p in zip(feature_names, feature_paths)]

    # specify nontemporal features to extract
    nontemporal_feature_paths = ["sql/year_of_birth.sql", 
                                 "sql/gender.sql",
                                 "sql/ethnicity_hispanic_or_latino.sql",
                                 "sql/race_american_indian_or_alaska_native.sql", 
                                 "sql/race_asian.sql",
                                 "sql/race_black_or_african_american.sql",
                                 "sql/race_native_hawaiian_or_other_pacific_islander.sql",
                                 "sql/race_white.sql"]
    nontemporal_feature_paths = [join(dirname(abspath(__file__)), path) 
                                 for path in nontemporal_feature_paths]
    nontemporal_feature_names = ["year_of_birth", 
                                 "gender",
                                 "ethnicity_hispanic_or_latino",
                                 "race_american_indian_or_alaska_native",
                                 "race_asian",
                                 "race_black_or_african_american",
                                 "race_native_hawaiian_or_other_pacific_islander",
                                 "race_white"]
    features.extend([Feature(n, p, temporal=False, params=feature_sql_params) 
                     for n, p in zip(nontemporal_feature_names, nontemporal_feature_paths)])

    # set up omop pkg dataset that does feature extraction
    omop_config = OMOPConfig({
        "path"          : "postgresql://" + config.db_name,
        "cdm_schema"    : "cdm",
        "aux_cdm_schema": "cdm_aux",
        "prefix_schema" : config.nonstationarity_schema_name
    })
    if os.path.exists(config.omop_data_dir + dataset_json_dir + 'data.json'):
        this_start_time = time.time()
        dataset         = OMOPDataset.from_prebuilt(name      = dataset_json_dir,
                                                    data_dir  = Path(config.omop_data_dir),
                                                    cache_dir = cache_path)
        logger.info('Time to load OMOP dataset from json: ' + str(time.time() - this_start_time) + ' seconds')
    else:
        this_start_time = time.time()
        dataset         = OMOPDataset(name      = dataset_json_dir, 
                                      config    = omop_config, 
                                      cohort    = cohort,
                                      features  = features,
                                      data_dir  = Path(config.omop_data_dir),
                                      cache_dir = cache_path)
        logger.info('Time to create OMOP dataset from scratch: ' + str(time.time() - this_start_time) + ' seconds')
    
    this_start_time = time.time()
    feature_matrix, times_list, person_ids = build_3d_sparse_feature_matrix(dataset.tokenizer,
                                                                            config.omop_data_dir + dataset_json_dir,
                                                                            logger)
    logger.info('Time to build 3-d sparse feature matrix: ' + str(time.time() - this_start_time) + ' seconds')
    
    this_start_time = time.time()
    save_coo_matrix_to_h5py(features_hf5_file,
                            feature_matrix)
    save_data_to_h5py(features_auxiliary_hf5_file,
                      {'times_list': times_list,
                       'person_ids': person_ids})
    logger.info('Time to save features: ' + str(time.time() - this_start_time) + ' seconds')
    return feature_matrix, times_list, person_ids
    
def extract_windowed_features(dataset_json_dir,
                              feature_matrix,
                              times_list,
                              window_days,
                              windowed_features_fileheader,
                              eligibility_time,
                              logger):
    '''
    At each time point, create windowed feature matrices for each feature type
    Read from dataset json (must be available)
    Write hf5 and json files for each feature type and date
    @param dataset_json_dir: str, name of directory where omop dataset json is stored
    @param feature_matrix: 3d sparse tensor
    @param time_list: list, dates for each index in feature_matrix
    @param window_days: list of ints, number of days in each feature window
    @param windowed_features_fileheader: str, start of path to files where windowed feature matrices are stored,
                                         append _general_date{idx}.hf5 and _general_feature_names.json 
                                                for gender, races, ethnicity, and predicted in month,
                                                _labs_date{idx}.hf5 and _labs_feature_names.json, 
                                                _conditions_date{idx}.hf5 and _conditions_feature_names.json, 
                                                _procedures_date{idx}.hf5 and _procedures_feature_names.json, 
                                                _drugs_date{idx}.hf5 and _drugs_feature_names.json, 
                                                _specialties_date{idx}.hf5 and _specialties_feature_names.json,
                                                _age_date{idx}.hf5 and _age_feature_names.json
    @param eligibility_time: str, require observed for at least 95% of this time prior to prediction date, 
                             e.g. 1 year or 3 years, {window} is same time with '_' replacing ' '
    @param logger: logger, for INFO messages
    @return: None
    '''
    assert len(times_list) == feature_matrix.shape[1]
    assert len(window_days) > 0
    if dataset_json_dir[-1] != '/':
        dataset_json_dir += '/'
    dataset_json_full_path = config.omop_data_dir + dataset_json_dir + 'data.json'
    assert os.path.exists(dataset_json_full_path), dataset_json_full_path + ' does not exist'
    cache_path             = Path(config.cache_dir)
    this_start_time        = time.time()
    dataset                = OMOPDataset.from_prebuilt(name      = dataset_json_dir,
                                                       data_dir  = Path(config.omop_data_dir),
                                                       cache_dir = cache_path)
    logger.info('Time to load dataset from json: ' + str(time.time() - this_start_time) + ' seconds')
    
    this_start_time          = time.time()
    feature_types            = ['age', 'general', 'labs', 'conditions', 'procedures', 'drugs', 'specialties']
    feat_type_map            = {'lab'      : 'labs',
                                'drug'     : 'drugs',
                                'condition': 'conditions',
                                'procedure': 'procedures',
                                'specialty': 'specialties'}
    feat_idxs_dict           = {feat_type: []
                                for feat_type in feature_types[2:]}
    feat_names_dict          = {feat_type: []
                                for feat_type in feature_types[2:]}
    for idx in range(3, len(dataset.tokenizer.concept_list)): # start at 3 to remove padding tokens
        if idx % 1000 == 0:
            gc.collect()
        name = dataset.tokenizer.concept_list[idx]
        feature_fields = name.split(' - ')
        assert len(feature_fields) >= 2
        assert feature_fields[1] in feat_type_map.keys()
        feat_idxs_dict[feat_type_map[feature_fields[1]]].append(idx)
        feat_names_dict[feat_type_map[feature_fields[1]]].append(name)
    if len(window_days) > 1:
        num_features = len(dataset.tokenizer.concept_list) - 3
        orig_num_features_per_type = {feat_type: len(feat_idxs_dict[feat_type])
                                      for feat_type in feat_idxs_dict}
        for day_idx in range(1, len(window_days)):
            for feat_type in feat_idxs_dict:
                feat_idxs_dict[feat_type] += [num_features * day_idx + feat_idxs_dict[feat_type][i] 
                                              for i in range(orig_num_features_per_type[feat_type])]
    logger.info('Time to categorize features: ' + str(time.time() - this_start_time) + ' seconds')
    
    this_start_time = time.time()
    feature_names_file_dict  = {feat_type: windowed_features_fileheader + '_'  + feat_type + '_feature_names.json'
                                for feat_type in feature_types}
    with open(feature_names_file_dict['age'], 'w') as f:
        json.dump(['age'], f)
        
    genders               = ['Male']
    ethnicities           = ['Hispanic or Latino']
    races                 = ['American Indian or Alaska Native', 'Asian', 'Black or African American', 
                             'Native Hawaiian or Other Pacific Islander', 'White']
    prediction_months     = ['pred_in_Jan', 'pred_in_Feb', 'pred_in_Mar', 'pred_in_Apr', 
                             'pred_in_May', 'pred_in_Jun', 'pred_in_Jul', 'pred_in_Aug', 
                             'pred_in_Sep', 'pred_in_Oct', 'pred_in_Nov', 'pred_in_Dec']
    general_feature_names = genders + ethnicities + races + prediction_months
    with open(feature_names_file_dict['general'], 'w') as f:
        json.dump(general_feature_names, f)
    for feat_type in feature_types[2:]:
        feat_type_names = []
        for days in window_days:
            feat_type_names += ['{} - {} days'.format(name, days) for name in feat_names_dict[feat_type]]
        with open(feature_names_file_dict[feat_type], 'w') as f:
            json.dump(feat_type_names, f)
        del feat_type_names
    gc.collect()
    logger.info('Time to save feature names: ' + str(time.time() - this_start_time) + ' seconds')
          
    this_start_time = time.time()
    features_fileheader_dict = {feat_type: windowed_features_fileheader + '_' + feat_type + '_date'
                                for feat_type in feature_types}
    dshf = dataset.to_hf()['train']
    gender_array        = np.array(dshf[genders[0]],
                                   dtype=np.int8).reshape((-1,1))
    ethnicity_array     = np.array(dshf[ethnicities[0]],
                                   dtype=np.int8).reshape((-1,1))
    race_arrays         = [np.array(dshf[race],
                                    dtype=np.int8).reshape((-1,1))
                           for race in races]
    year_of_birth_array = np.array(dshf['Year of birth']).reshape((-1,1))
    prediction_dates    = _get_prediction_dates(eligibility_time)
    all_times           = pd.to_datetime(np.array(from_unixtime(times_list)))
    logger.info('Time for remaining set-up before processing each prediction date: '
                 + str(time.time() - this_start_time) + ' seconds')
    
    for prediction_date_idx in range(len(prediction_dates)):
        date_start_time = time.time()
        
        this_start_time = time.time()
        prediction_date = prediction_dates[prediction_date_idx]
        age_array       = int(prediction_date[:4]) - year_of_birth_array
        save_data_to_h5py(features_fileheader_dict['age'] + str(prediction_date_idx) + '.hf5',
                          {'age': age_array})
        logger.info('Time to save age for prediction date ' + str(prediction_date_idx) + ': ' 
                    + str(time.time() - this_start_time) + ' seconds')
        del age_array
        
        this_start_time        = time.time()
        prediction_month_array = np.zeros((gender_array.shape[0],12), 
                                           dtype=np.int8)
        prediction_month_array[:,int(prediction_date[5:7])-1] = 1
        general_feature_matrix = csr_matrix(np.hstack((gender_array,
                                                       ethnicity_array,
                                                       *race_arrays,
                                                       prediction_month_array)),
                                            dtype=np.int8)
        save_sparse_matrix_to_h5py(features_fileheader_dict['general'] + str(prediction_date_idx) 
                                   + '.hf5',
                                   general_feature_matrix)
        del general_feature_matrix
        del prediction_month_array
        logger.info('Time to save general features for prediction date ' + str(prediction_date_idx) + ': ' 
                    + str(time.time() - this_start_time) + ' seconds')
        
        this_start_time         = time.time()
        window_end              = datetime.strptime(prediction_date, '%Y-%m-%d')
        window_end_idx          = all_times.searchsorted(window_end)
        feature_matrix_slices   = []
        for days in window_days:
            days_start_time     = time.time()
            window_start        = window_end - timedelta(days=days)
            window_start_idx    = all_times.searchsorted(window_start)
            feature_matrix_slices.append(np.clip(feature_matrix[:, window_start_idx:window_end_idx].sum(axis=1), 
                                                 0, 1).tocsr())
            logger.info('Time to append a window slice: ' + str(time.time() - days_start_time) + ' seconds')
        logger.info('Time to get all window slices: ' + str(time.time() - this_start_time) + ' seconds')
        this_start_time         = time.time()
        windowed_feature_matrix = hstack(feature_matrix_slices,
                                         format='csc',
                                         dtype=np.int8)
        del feature_matrix_slices
        logger.info('Time to create windowed feature matrix for prediction date ' + str(prediction_date_idx) + ': '
                    + str(time.time() - this_start_time) + ' seconds')
        
        for feat_type in feature_types[2:]:
            this_start_time              = time.time()
            feature_type_windowed_matrix = windowed_feature_matrix[:,feat_idxs_dict[feat_type]]
            logger.info('Time to get ' + feat_type + ' windowed features: ' 
                         + str(time.time() - this_start_time) + ' seconds')
            this_start_time              = time.time()
            save_sparse_matrix_to_h5py(features_fileheader_dict[feat_type] + str(prediction_date_idx) + '.hf5',
                                       feature_type_windowed_matrix)
            del feature_type_windowed_matrix
            logger.info('Time to save ' + feat_type + ' windowed features: ' 
                         + str(time.time() - this_start_time) + ' seconds')
        del windowed_feature_matrix
        gc.collect()
        logger.info('Finished ' + prediction_date + ' in ' + str(time.time() - date_start_time) + ' seconds')
    return
   
def extract_outcomes(outcome_sql_file,
                     outcome_sql_params,
                     dataset_json_dir,
                     eligible_ids,
                     outcomes_hf5_file,
                     eligibility_time,
                     logger):
    '''
    Extract list of outcomes for each prediction date
    Creates table condition_{outcome_id}_outcomes, procedure_{outcome_name}_outcomes, 
    lab_{outcome_id}_{direction}, or lab_group_{outcome_name}_outcomes in {nonstationarity_schema_name}
    @param outcome_sql_file: str, name of sql file for creating outcome table, filled in with outcome_sql_params
    @param outcome_sql_params: dict containing 1. outcome_id: str, id of condition, procedure, or measurement outcome
                                               2. outcome_name: str, name of procedure group
                                               3. direction: str, 'low' or 'high' if measurement outcome
                                               4. debug_suffix: str, '_debug{size}' or '' for whether to use limited cohort size
    @param dataset_json_dir: str, name of directory where omop dataset json is stored
    @param eligible_ids: list of int, person_id order in outcome arrays
    @param outcomes_hf5_file: str, path to file where outcomes_matrix is stored
    @param eligibility_time: str, require observed for at least 95% of this time prior to prediction date, 
                             e.g. 1 year or 3 years, {window} is same time with '_' replacing ' '
    @param logger: logger, for INFO messages
    @return: outcomes_matrix, csc matrix of people in cohort by prediction date
    '''
    if os.path.exists(outcomes_hf5_file):
        logger.info('Loading outcomes from ' + outcomes_hf5_file)
        return load_sparse_matrix_from_h5py(outcomes_hf5_file)
    
    if dataset_json_dir[-1] != '/':
        dataset_json_dir += '/'
    dataset_json_full_path = config.omop_data_dir + dataset_json_dir + 'data.json'
    cache_path             = Path(config.cache_dir)
    assert os.path.exists(dataset_json_full_path), dataset_json_full_path + ' does not exist'
    
    engine = config.create_sqlalchemy_engine()
        
    # create table if outcome is not eol
    if len(outcome_sql_file) > 0:
        outcome_sql_params = deepcopy(outcome_sql_params)
        assert 'outcome_id' in outcome_sql_params.keys()
        if 'procedure' in outcome_sql_file or 'lab_group' in outcome_sql_file:
            assert 'outcome_name' in outcome_sql_params.keys()
        if 'lab' in outcome_sql_file:
            assert 'direction' in outcome_sql_params.keys()
            assert outcome_sql_params['direction'] in {'low', 'high'}
            if outcome_sql_params['direction'] == 'low':
                outcome_sql_params['sign'] = '<'
            else:
                outcome_sql_params['sign'] = '>'
        outcome_sql_params['schema_name']  = config.nonstationarity_schema_name
        with open(join(dirname(abspath(__file__)), 'sql/' + outcome_sql_file), 'r') as f:
            outcome_table_creation_sql = f.read()
        outcome_table_creation_sql = outcome_table_creation_sql.format(**outcome_sql_params)
        if 'condition' in outcome_sql_file:
            outcome_table_name = 'condition_{outcome_id}_outcomes{debug_suffix}'.format(**outcome_sql_params)
        elif 'procedure' in outcome_sql_file:
            outcome_table_name = 'procedure_{outcome_name}_outcomes{debug_suffix}'.format(**outcome_sql_params)
        elif 'lab_group' in outcome_sql_file:
            outcome_table_name = 'lab_group_{outcome_name}_{direction}_outcomes{debug_suffix}'.format(**outcome_sql_params)
        else:
            assert 'lab' in outcome_sql_file
            outcome_table_name = 'lab_{outcome_id}_{direction}_outcomes{debug_suffix}'.format(**outcome_sql_params)
        with session_scope(engine) as session:
            session.execute(sqlalchemy.text(outcome_table_creation_sql))
            session.commit()
    else:
        outcome_table_name = 'monthly_eligibility_1_year'
        outcome_table_name = outcome_table_name.format(debug_suffix = outcome_sql_params['debug_suffix'])
    
    outcomes_sql = ('SELECT person_id, '
                           'end_date '
                    'FROM {schema_name}.{outcome_table_name} '
                    'WHERE y=1;'
                   )
    outcomes_sql = outcomes_sql.format(schema_name        = config.nonstationarity_schema_name,
                                       outcome_table_name = outcome_table_name)
    with session_scope(engine) as session:
        outcomes_result = session.execute(sqlalchemy.text(outcomes_sql))
        session.commit()
        
    prediction_dates = _get_prediction_dates(eligibility_time)
    prediction_dates_dict = {prediction_dates[i]: i for i in range(len(prediction_dates))}
    eligible_ids_dict = {eligible_ids[i]: i for i in range(len(eligible_ids))}
    outcomes_matrix = np.zeros((len(eligible_ids), len(prediction_dates)))
    for row in outcomes_result:
        outcomes_matrix[eligible_ids_dict[row['person_id']], prediction_dates_dict[str(row['end_date'])]] = 1
        
    dataset = OMOPDataset.from_prebuilt(name      = dataset_json_dir,
                                        data_dir  = Path(config.omop_data_dir),
                                        cache_dir = cache_path)
    dshf = dataset.to_hf()['train']
    outcomes_matrix = outcomes_matrix[dshf['cohort_id']]
    outcomes_matrix = csc_matrix(outcomes_matrix, dtype=np.int8)
    save_sparse_matrix_to_h5py(outcomes_hf5_file,
                               outcomes_matrix)
    return outcomes_matrix
    
def split_data(eligible_ids, 
               outcome_table_name,
               n_folds,
               include_test_split,
               data_split_hf5_header,
               eligibility_time,
               logger):
    '''
    Split patients who have outcome for the first time at each time point into n-fold cross validation
    and split patients who never have outcome into n-fold cross validation
    Option to include test set if training and testing happen on the same time point
    @param eligible_ids: list of int, person ids to split
    @param outcome_table_name: str, name of table containing outcomes, expects table in schema,
                               expects columns person_id, y, end_date
    @param n_folds: int, number of folds for cross-validation
    @param include_test_split: boolean, whether to include a test split
    @param data_split_hf5_header: str, path to store data split indices, will append _train.hf5, _valid.hf5, _test.hf5
    @param eligibility_time: str, require observed for at least 95% of this time prior to prediction date, 
                             e.g. 1 year or 3 years, {window} is same time with '_' replacing ' '
    @param logger: logger, for INFO messages
    @return: 2 lists of np arrays, binary indicators for whether each sample is in training or validation set of each fold
             if include_test_split, an array of indicators for test set is also returned
    '''
    train_data_split_h5_file = data_split_hf5_header + '_train.hf5'
    valid_data_split_h5_file = data_split_hf5_header + '_valid.hf5'
    test_data_split_h5_file  = data_split_hf5_header + '_test.hf5'
    if  os.path.exists(train_data_split_h5_file) \
    and os.path.exists(valid_data_split_h5_file) \
    and ((not include_test_split) or os.path.exists(test_data_split_h5_file)):
        loading_files_str = train_data_split_h5_file + ', ' + valid_data_split_h5_file
        if include_test_split:
            loading_files_str += ', ' + test_data_split_h5_file
        logger.info('Loading fold indicators from ' + loading_files_str)
        fold_train_indicators_dict = load_data_from_h5py(train_data_split_h5_file)
        fold_train_indicators      = [fold_train_indicators_dict['fold' + str(fold_idx)] for fold_idx in range(n_folds)]
        fold_valid_indicators_dict = load_data_from_h5py(valid_data_split_h5_file)
        fold_valid_indicators      = [fold_valid_indicators_dict['fold' + str(fold_idx)] for fold_idx in range(n_folds)]
        if include_test_split:
            test_indicators        = load_data_from_h5py(test_data_split_h5_file)['test_indicators']
            return fold_train_indicators, fold_valid_indicators, test_indicators
        return fold_train_indicators, fold_valid_indicators
    
    assert n_folds > 0
    np.random.seed(1807)
    fold_train_idxs = [[] for i in range(n_folds)]
    fold_valid_idxs = [[] for i in range(n_folds)]
    if include_test_split:
        test_idxs = []
    eligible_ids_dict = {eligible_ids[i]: i for i in range(len(eligible_ids))}
    # n-fold cross-validation among patients who have outcome for each month 
    # (last month for patients who have outcome multiple times)
    prediction_dates = _get_prediction_dates(eligibility_time)
    engine = config.create_sqlalchemy_engine()
    with session_scope(engine) as session:
        for prediction_date in prediction_dates:
            outcome1_sql = ('WITH outcome1_before_cohort AS ( '
                                'SELECT DISTINCT person_id '
                                'FROM {schema_name}.{outcome_table_name} '
                                'WHERE y = 1 '
                                'AND end_date < DATE(\'{prediction_date}\')) '
                            'SELECT DISTINCT ot.person_id '
                            'FROM {schema_name}.{outcome_table_name} ot '
                            'WHERE ot.y = 1 '
                            'AND ot.end_date = DATE(\'{prediction_date}\') '
                            'AND NOT EXISTS ( '
                                'SELECT '
                                'FROM outcome1_before_cohort obc '
                                'WHERE obc.person_id = ot.person_id '
                            ') '
                            'ORDER BY ot.person_id;'
                           )
            outcome1_sql = outcome1_sql.format(schema_name        = config.nonstationarity_schema_name,
                                               outcome_table_name = outcome_table_name,
                                               prediction_date    = prediction_date)
            outcome1_results = session.execute(sqlalchemy.text(outcome1_sql))
            session.commit()
            outcome1_ids = [row['person_id'] for row in outcome1_results]
            outcome1_idxs = [eligible_ids_dict[person_id] for person_id in outcome1_ids if person_id in eligible_ids_dict]
            if len(outcome1_idxs) < n_folds + 1:
                # Too few samples to split, so place all in train
                for fold_idx in range(n_folds):
                    fold_train_idxs[fold_idx].extend(outcome1_idxs)
                continue
            if include_test_split:
                outcome1_train_valid_idxs, outcome1_test_idxs = train_test_split(outcome1_idxs, 
                                                                                 test_size=.2, 
                                                                                 random_state=0)
                test_idxs.extend(outcome1_test_idxs)
            else:
                outcome1_train_valid_idxs = outcome1_idxs
            
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
            fold_idx = 0
            for train_indices, valid_indices in kf.split(outcome1_train_valid_idxs):
                fold_train_idxs[fold_idx].extend(np.array(outcome1_train_valid_idxs)[train_indices].tolist())
                fold_valid_idxs[fold_idx].extend(np.array(outcome1_train_valid_idxs)[valid_indices].tolist())
                fold_idx += 1
            
        # n-fold cross-validation among patients who never have outcome
        outcome0_sql = ('WITH outcome1_cohort AS ( '
                            'SELECT DISTINCT person_id '
                            'FROM {schema_name}.{outcome_table_name} '
                            'WHERE y = 1) '
                        'SELECT DISTINCT ot.person_id '
                        'FROM {schema_name}.{outcome_table_name} ot '
                        'WHERE NOT EXISTS ( '
                            'SELECT oc.person_id '
                            'FROM outcome1_cohort oc '
                            'WHERE oc.person_id = ot.person_id) '
                        'ORDER BY ot.person_id;'
                       )
        outcome0_sql = outcome0_sql.format(schema_name        = config.nonstationarity_schema_name,
                                           outcome_table_name = outcome_table_name)
        outcome0_results = session.execute(sqlalchemy.text(outcome0_sql))
        session.commit()
        outcome0_ids = [row['person_id'] for row in outcome0_results]
    outcome0_idxs = [eligible_ids_dict[person_id] for person_id in outcome0_ids if person_id in eligible_ids_dict]
    if include_test_split:
        outcome0_train_valid_idxs, outcome0_test_idxs = train_test_split(outcome0_idxs, 
                                                                         test_size=.2, 
                                                                         random_state=0)
        test_idxs.extend(outcome0_test_idxs)
    else:
        outcome0_train_valid_idxs = outcome0_idxs
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    fold_idx = 0
    for train_indices, valid_indices in kf.split(outcome0_train_valid_idxs):
        fold_train_idxs[fold_idx].extend(np.array(outcome0_train_valid_idxs)[train_indices].tolist())
        fold_valid_idxs[fold_idx].extend(np.array(outcome0_train_valid_idxs)[valid_indices].tolist())
        fold_idx += 1
        
    # turn into binary indicators
    fold_train_indicators = [np.zeros(len(eligible_ids)) for i in range(n_folds)]
    fold_valid_indicators = [np.zeros(len(eligible_ids)) for i in range(n_folds)]
    for fold_idx in range(n_folds):
        fold_train_indicators[fold_idx][fold_train_idxs[fold_idx]] = 1
        fold_valid_indicators[fold_idx][fold_valid_idxs[fold_idx]] = 1
    if include_test_split:
        test_indicators = np.zeros(len(eligible_ids))
        test_indicators[test_idxs] = 1
    save_data_to_h5py(train_data_split_h5_file,
                      {'fold' + str(fold_idx): fold_train_indicators[fold_idx] for fold_idx in range(n_folds)})
    save_data_to_h5py(valid_data_split_h5_file,
                      {'fold' + str(fold_idx): fold_valid_indicators[fold_idx] for fold_idx in range(n_folds)})
    if include_test_split:
        save_data_to_h5py(test_data_split_h5_file,
                          {'test_indicators': test_indicators})
        return fold_train_indicators, fold_valid_indicators, test_indicators
    return fold_train_indicators, fold_valid_indicators

def reindex_monthly_eligibility(monthly_eligibility,
                                dataset_json_dir,
                                reindexed_monthly_eligibility_hf5_file,
                                logger):
    '''
    Re-index monthly eligibility matrix to match cohort
    @param monthly_eligibility: np array, # people x # prediction dates
    @param dataset_json_dir: str, name of directory where omop dataset json is stored
    @param reindexed_monthly_eligibility_hf5_file: str, path to file to save reindexed_monthly_eligibility. If exists, load.
    @param logger: logger, for INFO messages
    @return: reindexed_monthly_eligibility, np array, # people in cohort x # prediction dates
    '''
    if os.path.exists(reindexed_monthly_eligibility_hf5_file):
        logger.info('Loading reindexed monthly eligibility from ' + reindexed_monthly_eligibility_hf5_file)
        return load_data_from_h5py(reindexed_monthly_eligibility_hf5_file)['monthly_eligibility']
    cache_path      = Path(config.cache_dir)
    this_start_time = time.time()
    dataset = OMOPDataset.from_prebuilt(name      = dataset_json_dir,
                                        data_dir  = Path(config.omop_data_dir),
                                        cache_dir = cache_path)
    dshf = dataset.to_hf()['train']
    logger.info('Time to load dataset from json: ' + str(time.time() - this_start_time) + ' seconds')
    reindexed_monthly_eligibility = monthly_eligibility[dshf['cohort_id']]
    save_data_to_h5py(reindexed_monthly_eligibility_hf5_file,
                      {'monthly_eligibility': reindexed_monthly_eligibility})
    return reindexed_monthly_eligibility

def extract_condition_monthly_eligibility(eligible_ids,
                                          outcome_table_name,
                                          condition_monthly_eligibility_hf5_file,
                                          dataset_json_dir,
                                          eligibility_time,
                                          logger):
    '''
    Extract monthly eligibility matrix for condition-outcome cohort
    Condition outcome table excludes patients who have had the condition before
    @param eligible_ids: list of int, order of person_ids
    @param outcome_table_name: str, name of table with person_id, end_date, 1 rows per eligible person-month, 
                               exists in {nonstationarity_schema_name}
    @param condition_monthly_eligibility_hf5_file: str, path to store array, read from if exists
    @param dataset_json_dir: str, name of directory where omop dataset json is stored
    @param eligibility_time: str, require observed for at least 95% of this time prior to prediction date, 
                             e.g. 1 year or 3 years, {window} is same time with '_' replacing ' '
    @param logger: logger, for INFO messages
    @return condition_monthly_eligibility: np array of months where patient is eligible in condition outcome table
    '''
    if os.path.exists(condition_monthly_eligibility_hf5_file):
        logger.info('Loading condition monthly eligibility from ' + condition_monthly_eligibility_hf5_file)
        return load_data_from_h5py(condition_monthly_eligibility_hf5_file)['monthly_eligibility']
    
    engine = config.create_sqlalchemy_engine()
    with session_scope(engine) as session:
        
        # map prediction date to column index of array
        prediction_dates = _get_prediction_dates(eligibility_time)
        prediction_dates_dict = {date: i for i, date in enumerate(prediction_dates)}
        
        # map person_id to row index of array
        eligible_ids_dict = {person_id: i for i, person_id in enumerate(eligible_ids)}

        # create array of patients x prediction dates
        # binary indicators for whether patient is eligible that month
        condition_monthly_eligibility = np.zeros((len(eligible_ids), len(prediction_dates)), dtype=np.int8)
        select_eligible_sql = ('SELECT person_id, '
                                      'end_date '
                               'FROM {schema_name}.{outcome_table_name} '
                               'ORDER BY person_id, '
                                        'end_date;'
                              )
        select_eligible_sql = select_eligible_sql.format(schema_name        = config.nonstationarity_schema_name,
                                                         outcome_table_name = outcome_table_name)
        select_eligible_result = session.execute(sqlalchemy.text(select_eligible_sql))
        session.commit()
        
        for row in select_eligible_result:
            eligible_id_idx     = eligible_ids_dict[row['person_id']]
            prediction_date_idx = prediction_dates_dict[row['end_date'].strftime('%Y-%m-%d')]
            condition_monthly_eligibility[eligible_id_idx, prediction_date_idx] = 1
            
        return reindex_monthly_eligibility(condition_monthly_eligibility,
                                           dataset_json_dir,
                                           condition_monthly_eligibility_hf5_file,
                                           logger)
    
def finalize_data(windowed_features_fileheader, 
                  outcomes_matrix, 
                  monthly_eligibility,
                  train_indicators, 
                  valid_indicators, 
                  test_indicators,
                  min_freq,
                  output_file_header,
                  logger):
    '''
    Create a sequence of sets of training, validation, and test features and outcomes for the specified fold.
    Sequence has one dataset per year, so merge data from all months of year and shuffle.
    Create many feature sets: 1. age, 2. non-temporal features, 3. labs, 4. conditions, 5. procedures, 6. drugs, 
                              7. specialties
    Select features with minimum frequency in training set across all years.
    Save to json and npz files. Will overwrite.
    @param windowed_features_fileheader: str, start of path to files where windowed feature matrices are stored,
                                         append _{feat_type}_date{idx}.hf5 and _{feat_type}_feature_names.json
    @param outcomes_matrix: sparse matrix, outcomes at each prediction date by month,
                            assumes every 12 columns is a year, if 72 columns: 2015-2020, if 48 columns: 2017-2020
                            throws error if other number of columns
    @param monthly_eligibility: array, # patients x # months whether patient is eligible each month
                                assumes same number of columns as outcomes_matrix
    @param train_indicators: array, binary indicators for whether patient is in training set
    @param valid_indicators: array, binary indicators for whether patient is in validation set
    @param test_indicators: array, binary indicators for whether patient is in test set
    @param min_freq: int, minimum # patients in training set to include feature
    @param output_file_header: str, start of file path to store datasets as dicts with train, valid, test, feature_names
                               will append _general_{data} (for gender and predicted in month), _labs_{data}, 
                               _conditions_{data}, _procedures_{data}, _drugs_{data}, _specialties_{data},
                               _age_{data}, _outcomes_{data}
                               where {data} is train_year{idx}.hf5, valid_year{idx}.hf5, test_year{idx}.hf5, 
                               feature_names.json
    @param logger: logger, for INFO messages
    @return: dict mapping str of data split to list of np arrays, outcomes for each year
    '''
    start_time                        = time.time()
    this_start_time                   = time.time()
    assert outcomes_matrix.shape[1] == 72 or outcomes_matrix.shape[1] == 48
    assert outcomes_matrix.shape[1] == monthly_eligibility.shape[1]
    num_years                         = int(outcomes_matrix.shape[1] / 12) # years: 2015 - 2020 or 2017 - 2020
    data_splits                       = ['train', 'valid', 'test']
    feature_types                     = ['age', 'general', 'labs', 'conditions', 'procedures', 'drugs', 'specialties']
    features_file_dict                = {feat_type: {data_split: [output_file_header + '_' + feat_type + '_' + data_split 
                                                                  + '_year' + str(year_idx) + '.hf5'
                                                                  for year_idx in range(num_years)]
                                                     for data_split in data_splits}
                                         for feat_type in feature_types}
    feature_names_file_dict           = {feat_type: output_file_header + '_'  + feat_type + '_feature_names.json'
                                         for feat_type in feature_types}
    outcomes_file_dict                = {data_split: [output_file_header + '_outcomes_' + data_split 
                                                      + '_year' + str(year_idx) + '.hf5'
                                                      for year_idx in range(num_years)]
                                         for data_split in data_splits}
    windowed_features_fileheader_dict = {feat_type: windowed_features_fileheader + '_' + feat_type + '_date'
                                         for feat_type in feature_types}
    orig_feature_names_file_dict      = {feat_type: windowed_features_fileheader + '_' + feat_type + '_feature_names.json'
                                         for feat_type in feature_types}
    indicators_dict                   = {'train': train_indicators,
                                         'valid': valid_indicators,
                                         'test' : test_indicators}
    logger.info('Time to set up finalize_data: ' + str(time.time() - this_start_time) + ' seconds')
    
    # split data for each month
    this_start_time             = time.time()
    split_idxs_dict           = {data_split: dict() for data_split in data_splits}
    num_samples_per_year_dict = {data_split: np.zeros(num_years) for data_split in data_splits}
    for year_idx in range(num_years):
        for month_idx in range(12*year_idx, 12*(year_idx+1)):
            for data_split in data_splits:
                data_split_idxs = np.nonzero(np.where(np.logical_and(indicators_dict[data_split]==1,
                                                                     monthly_eligibility[:,month_idx]==1), 1, 0))[0]
                split_idxs_dict[data_split][month_idx] = data_split_idxs
                num_samples_per_year_dict[data_split][year_idx] += len(data_split_idxs)
    del monthly_eligibility
    del indicators_dict
    logger.info('# samples in train, valid, test')
    if num_years == 6:
        starting_year = 2015
    else:
        starting_year = 2017
    for year_idx in range(num_years):
        logger.info(str(starting_year + year_idx) + ': ' + str(num_samples_per_year_dict['train'][year_idx]) + ', ' 
                    + str(num_samples_per_year_dict['valid'][year_idx]) + ', '
                    + str(num_samples_per_year_dict['test'][year_idx]))
    logger.info('Time to get data split indices for each month: ' + str(time.time() - this_start_time) + ' seconds')
    
    # get indices for shuffling data so not ordered by month
    this_start_time   = time.time()
    shuffle_idxs_dict = {data_split: [np.arange(num_samples_per_year_dict[data_split][t],
                                                dtype=np.int) 
                                      for t in range(num_years)]
                         for data_split in data_splits}
    np.random.seed(1807)
    for data_split, t in product(data_splits, range(num_years)):
        np.random.shuffle(shuffle_idxs_dict[data_split][t])
        shuffle_idxs_dict[data_split][t] = shuffle_idxs_dict[data_split][t].tolist()
    logger.info('Time to shuffle data for each year: ' + str(time.time() - this_start_time) + ' seconds')
    
    this_start_time  = time.time()
    Y_all_years_dict = {data_split: [] for data_split in data_splits}
    for year_idx in range(num_years):
        year_Y_dict  = {data_split: [] for data_split in data_splits}
        for month_idx in range(12*year_idx, 12*(year_idx+1)):
            month_outcomes = outcomes_matrix[:,month_idx].toarray()
            for data_split in data_splits:
                year_Y_dict[data_split].append(month_outcomes[split_idxs_dict[data_split][month_idx]])
        for data_split in data_splits:
            year_split_outcomes = np.concatenate(year_Y_dict[data_split],
                                                 dtype=np.int8)[shuffle_idxs_dict[data_split][year_idx]]
            save_data_to_h5py(outcomes_file_dict[data_split][year_idx],
                              {'outcomes': year_split_outcomes})
            Y_all_years_dict[data_split].append(year_split_outcomes)
        gc.collect()
    logger.info('Time to concatenate, shuffle, and save outcomes: ' + str(time.time() - this_start_time) + ' seconds')
                
    for feat_type in feature_types:
        all_years_X_dict = {data_split: [] for data_split in data_splits}
        for year_idx in range(num_years):
            year_X_dict  = {data_split: [] for data_split in data_splits}
            for month_idx in range(12*year_idx, 12*(year_idx+1)):
                
                this_start_time = time.time()
                month_windowed_features_hf5_file = windowed_features_fileheader_dict[feat_type] + str(month_idx) + '.hf5'
                if feat_type == 'age':
                    month_windowed_features = load_data_from_h5py(month_windowed_features_hf5_file)['age']
                else:
                    month_windowed_features = load_sparse_matrix_from_h5py(month_windowed_features_hf5_file)
                for data_split in data_splits:
                    year_X_dict[data_split].append(month_windowed_features[split_idxs_dict[data_split][month_idx]])
                logger.info('Time to load a month of ' + feat_type + ' data: ' 
                            + str(time.time() - this_start_time) + ' seconds')
                
            this_start_time = time.time()
            for data_split in data_splits:
                if feat_type == 'age':
                    all_years_X_dict[data_split].append(np.vstack(year_X_dict[data_split])
                                                        [shuffle_idxs_dict[data_split][year_idx]])
                else:
                    all_years_X_dict[data_split].append(vstack(year_X_dict[data_split], 
                                                               format='csr', 
                                                               dtype=np.int8)
                                                        [shuffle_idxs_dict[data_split][year_idx]])
            logger.info('Time to stack and shuffle a year of ' + feat_type + ' data: '
                        + str(time.time() - this_start_time) + ' seconds')
            
        if feat_type == 'age':
            # scale age
            this_start_time = time.time()
            train_age_all = np.vstack(all_years_X_dict['train'])
            scaler = StandardScaler()
            scaler.fit(train_age_all)
            del train_age_all
            for data_split, year_idx in product(data_splits, range(num_years)):
                if len(all_years_X_dict[data_split][year_idx]) > 0:
                    all_years_X_dict[data_split][year_idx] = scaler.transform(all_years_X_dict[data_split][year_idx])
            with open(feature_names_file_dict[feat_type], 'w') as f:
                json.dump(['age'], f)
            age_scaler_file_name = output_file_header + '_age_scaling_params.txt'
            with open(age_scaler_file_name, 'w') as f:
                f.write(str(scaler.mean_[0]) + ',' + str(scaler.scale_[0]))
            logger.info('Time to scale age: ' + str(time.time() - this_start_time) + ' seconds')
        elif feat_type == 'general':
            with open(orig_feature_names_file_dict[feat_type], 'r') as f:
                general_feature_names = json.load(f)
            with open(feature_names_file_dict[feat_type], 'w') as f:
                json.dump(general_feature_names, f)
        else:
            # get frequent features
            this_start_time = time.time()
            train_freq_all = np.zeros(all_years_X_dict['train'][0].shape[1])
            for t in range(len(all_years_X_dict['train'])):
                train_freq_all += np.asarray(all_years_X_dict['train'][t].sum(axis=0)).flatten()
            freq_feat_idxs = np.nonzero(train_freq_all >= min_freq)[0]
            del train_freq_all    
            logger.info(str(all_years_X_dict['train'][0].shape[1]) + ' features')
            logger.info(str(len(freq_feat_idxs)) + ' frequent features')
            with open(orig_feature_names_file_dict[feat_type], 'r') as f:
                feature_names = json.load(f)
            freq_feature_names = []
            for idx in freq_feat_idxs:
                freq_feature_names.append(feature_names[idx])
            with open(feature_names_file_dict[feat_type], 'w') as f:
                json.dump(freq_feature_names, f)
            for data_split, year_idx in product(data_splits, range(num_years)):
                all_years_X_dict[data_split][year_idx] = all_years_X_dict[data_split][year_idx].tocsc()[:,freq_feat_idxs]
            logger.info('Time to get frequent features: ' + str(time.time() - this_start_time) + ' seconds')
        
        # save per year, split
        this_start_time = time.time()
        for data_split, year_idx in product(data_splits, range(num_years)):
            if feat_type == 'age':
                save_data_to_h5py(features_file_dict[feat_type][data_split][year_idx],
                                  {'age': all_years_X_dict[data_split][year_idx]})
            else:
                save_sparse_matrix_to_h5py(features_file_dict[feat_type][data_split][year_idx],
                                           all_years_X_dict[data_split][year_idx])
        logger.info('Time to save ' + feat_type + ' data: ' + str(time.time() - this_start_time) + ' seconds')
    logger.info('Time to run finalize_data: ' + str(time.time() - start_time) + ' seconds')
    return Y_all_years_dict

def get_person_id_to_sample_mapping(person_ids,
                                    monthly_eligibility,
                                    train_indicators,
                                    valid_indicators,
                                    test_indicators,
                                    output_file_header,
                                    logger):
    '''
    Using the given sample split and random seed used to finalize data, compute the sample indices for each patient ID
    in each year and data split
    @param person_ids: list of int, length: # patients, person IDs
    @param monthly_eligibility: np array of ints, # patients x # months, 
                                binary indicators for whether patient is eligible each month
    @param train_indicators: np array of ints, binary indicators for whether patient is in training set
    @param valid_indicators: np array of ints, binary indicators for whether patient is in validation set
    @param test_indicators: np array of ints, binary indicators for whether patient is in test set
    @param output_file_header: str, start of file path to store dict mapping person ID to sample indices 
                               in that year/data split, will append _person_ids_{data} 
                               where {data} is train_year{idx}.json, valid_year{idx}.json, test_year{idx}.json
    @param logger: logger, for INFO messages
    @return: None
    '''
    start_time      = time.time()
    this_start_time = time.time()
    
    person_ids      = np.array(person_ids)
    num_people      = len(person_ids)
    
    assert monthly_eligibility.shape[0] == num_people
    assert monthly_eligibility.shape[1] == 72 or monthly_eligibility.shape[1] == 48
    assert len(train_indicators)        == num_people
    assert len(valid_indicators)        == num_people
    assert len(test_indicators)         == num_people
    
    num_years       = int(monthly_eligibility.shape[1]/12) # years: 2015 - 2020 or 2017 - 2020
    data_splits     = ['train', 'valid', 'test']
    indicators_dict = {'train': train_indicators,
                       'valid': valid_indicators,
                       'test' : test_indicators}
    
    # split data for each month
    split_idxs_dict           = {data_split: dict() for data_split in data_splits}
    num_samples_per_year_dict = {data_split: np.zeros(num_years) for data_split in data_splits}
    for year_idx in range(num_years):
        for month_idx in range(12*year_idx, 12*(year_idx+1)):
            for data_split in data_splits:
                data_split_idxs = np.nonzero(np.where(np.logical_and(indicators_dict[data_split]==1,
                                                                     monthly_eligibility[:,month_idx]==1), 1, 0))[0]
                split_idxs_dict[data_split][month_idx] = data_split_idxs
                num_samples_per_year_dict[data_split][year_idx] += len(data_split_idxs)
    del monthly_eligibility
    del indicators_dict
    logger.info('# samples in train, valid, test')
    if num_years == 6:
        starting_year = 2015
    else:
        starting_year = 2017
    for year_idx in range(num_years):
        logger.info(str(starting_year + year_idx) + ': ' + str(num_samples_per_year_dict['train'][year_idx]) + ', ' 
                    + str(num_samples_per_year_dict['valid'][year_idx]) + ', '
                    + str(num_samples_per_year_dict['test'][year_idx]))
    logger.info('Time to get data split indices for each month: ' + str(time.time() - this_start_time) + ' seconds')
    
    # get indices for shuffling data so not ordered by month
    this_start_time   = time.time()
    shuffle_idxs_dict = {data_split: [np.arange(num_samples_per_year_dict[data_split][t],
                                                dtype=np.int) 
                                      for t in range(num_years)]
                         for data_split in data_splits}
    np.random.seed(1807)
    for data_split, t in product(data_splits, range(num_years)):
        np.random.shuffle(shuffle_idxs_dict[data_split][t])
        shuffle_idxs_dict[data_split][t] = shuffle_idxs_dict[data_split][t].tolist()
    logger.info('Time to shuffle data for each year and data split: ' + str(time.time() - this_start_time) + ' seconds')
    
    # create person to sample indices mapping
    this_start_time = time.time()
    for data_split, year_idx in product(data_splits, range(num_years)):
        year_split_person_ids = np.concatenate([person_ids[split_idxs_dict[data_split][12 * year_idx + month_idx]]
                                                for month_idx in range(12)])
        year_split_person_ids = year_split_person_ids[shuffle_idxs_dict[data_split][year_idx]]
        person_id_to_sample_idx_dict = defaultdict(list)
        for sample_idx in range(len(year_split_person_ids)):
            person_id = int(year_split_person_ids[sample_idx])
            person_id_to_sample_idx_dict[person_id].append(sample_idx)
        json_filename = output_file_header + '_person_ids_' + data_split + '_year' + str(year_idx) + '.json'
        with open(json_filename, 'w') as f:
            json.dump(person_id_to_sample_idx_dict, f)
    logger.info('Time to save person to sample indices mapping for each year and data split: ' 
                + str(time.time() - this_start_time) + ' seconds')
    logger.info('Time to compute person to sample indices mappings: ' + str(time.time() - start_time) + ' seconds')
    
def gather_fold_data_with_other_fold_features(windowed_features_fileheader, 
                                              outcomes_matrix, 
                                              monthly_eligibility,
                                              train_indicators, 
                                              valid_indicators, 
                                              original_fold_file_header,
                                              output_file_header,
                                              logger):
    '''
    Gather data for a training and validation fold
    with features defined by frequency in another training fold
    Create a sequence of sets of training and validation features and outcomes for the specified fold.
    Sequence has one dataset per year, so merge data from all months of year and shuffle.
    Create many feature sets: 1. age, 2. non-temporal features, 3. labs, 4. conditions, 5. procedures, 6. drugs, 
                              7. specialties
    Save to hf5 files. Will overwrite.
    @param windowed_features_fileheader: str, start of path to files where windowed feature matrices are stored,
                                         append _{feat_type}_date{idx}.hf5 and _{feat_type}_feature_names.json
    @param outcomes_matrix: sparse matrix, outcomes at each prediction date by month,
                            assumes every 12 columns is a year, if 72 columns: 2015-2020, if 48 columns: 2017-2020
                            throws error if other number of columns
    @param monthly_eligibility: array, # patients x # months whether patient is eligible each month
                                assumes same number of columns as outcomes_matrix
    @param train_indicators: array, binary indicators for whether patient is in training set
    @param valid_indicators: array, binary indicators for whether patient is in validation set
    @param original_fold_file_header: str, start of file path containing feature names in original fold
    @param output_file_header: str, start of file path to store datasets as dicts with train, valid, test, feature_names
                               will append _general_{data} (for gender and predicted in month), _labs_{data}, 
                               _conditions_{data}, _procedures_{data}, _drugs_{data}, _specialties_{data},
                               _age_{data}, _outcomes_{data}
                               where {data} is train_year{idx}.hf5, valid_year{idx}.hf5
    @param logger: logger, for INFO messages
    @return: None
    '''
    start_time                        = time.time()
    this_start_time                   = time.time()
    assert outcomes_matrix.shape[1] == 72 or outcomes_matrix.shape[1] == 48
    assert outcomes_matrix.shape[1] == monthly_eligibility.shape[1]
    num_years                         = int(outcomes_matrix.shape[1] / 12) # years: 2015 - 2020 or 2017 - 2020
    data_splits                       = ['train', 'valid']
    feature_types                     = ['age', 'general', 'labs', 'conditions', 'procedures', 'drugs', 'specialties']
    features_file_dict                = {feat_type: {data_split: [output_file_header + '_' + feat_type + '_' + data_split 
                                                                  + '_year' + str(year_idx) + '.hf5'
                                                                  for year_idx in range(num_years)]
                                                     for data_split in data_splits}
                                         for feat_type in feature_types}
    orig_fold_feature_names_file_dict = {feat_type: original_fold_file_header + '_'  + feat_type + '_feature_names.json'
                                         for feat_type in feature_types}
    outcomes_file_dict                = {data_split: [output_file_header + '_outcomes_' + data_split 
                                                      + '_year' + str(year_idx) + '.hf5'
                                                      for year_idx in range(num_years)]
                                         for data_split in data_splits}
    windowed_features_fileheader_dict = {feat_type: windowed_features_fileheader + '_' + feat_type + '_date'
                                         for feat_type in feature_types}
    orig_feature_names_file_dict      = {feat_type: windowed_features_fileheader + '_' + feat_type + '_feature_names.json'
                                         for feat_type in feature_types}
    indicators_dict                   = {'train': train_indicators,
                                         'valid': valid_indicators}
    logger.info('Time to set up gather_fold_data_with_other_fold_features: ' + str(time.time() - this_start_time) + ' seconds')
    
    # split data for each month
    this_start_time             = time.time()
    split_idxs_dict           = {data_split: dict() for data_split in data_splits}
    num_samples_per_year_dict = {data_split: np.zeros(num_years) for data_split in data_splits}
    for year_idx in range(num_years):
        for month_idx in range(12*year_idx, 12*(year_idx+1)):
            for data_split in data_splits:
                data_split_idxs = np.nonzero(np.where(np.logical_and(indicators_dict[data_split]==1,
                                                                     monthly_eligibility[:,month_idx]==1), 1, 0))[0]
                split_idxs_dict[data_split][month_idx] = data_split_idxs
                num_samples_per_year_dict[data_split][year_idx] += len(data_split_idxs)
    del monthly_eligibility
    del indicators_dict
    logger.info('# samples in train, valid')
    if num_years == 6:
        starting_year = 2015
    else:
        starting_year = 2017
    for year_idx in range(num_years):
        logger.info(str(starting_year + year_idx) + ': ' + str(num_samples_per_year_dict['train'][year_idx]) + ', ' 
                    + str(num_samples_per_year_dict['valid'][year_idx]))
    logger.info('Time to get data split indices for each month: ' + str(time.time() - this_start_time) + ' seconds')
    
    # get indices for shuffling data so not ordered by month
    this_start_time   = time.time()
    shuffle_idxs_dict = {data_split: [np.arange(num_samples_per_year_dict[data_split][t],
                                                dtype=np.int) 
                                      for t in range(num_years)]
                         for data_split in data_splits}
    np.random.seed(4301)
    for data_split, t in product(data_splits, range(num_years)):
        np.random.shuffle(shuffle_idxs_dict[data_split][t])
        shuffle_idxs_dict[data_split][t] = shuffle_idxs_dict[data_split][t].tolist()
    logger.info('Time to shuffle data for each year: ' + str(time.time() - this_start_time) + ' seconds')
    
    this_start_time  = time.time()
    for year_idx in range(num_years):
        year_Y_dict  = {data_split: [] for data_split in data_splits}
        for month_idx in range(12*year_idx, 12*(year_idx+1)):
            month_outcomes = outcomes_matrix[:,month_idx].toarray()
            for data_split in data_splits:
                year_Y_dict[data_split].append(month_outcomes[split_idxs_dict[data_split][month_idx]])
        for data_split in data_splits:
            year_split_outcomes = np.concatenate(year_Y_dict[data_split],
                                                 dtype=np.int8)[shuffle_idxs_dict[data_split][year_idx]]
            save_data_to_h5py(outcomes_file_dict[data_split][year_idx],
                              {'outcomes': year_split_outcomes})
        gc.collect()
    logger.info('Time to concatenate, shuffle, and save outcomes: ' + str(time.time() - this_start_time) + ' seconds')
    
    # gather original age scaling parameters
    age_scaler_file_name = original_fold_file_header + '_age_scaling_params.txt'
    with open(age_scaler_file_name, 'r') as f:
        age_params       = f.read().strip().split(',')
    age_mean             = float(age_params[0])
    age_std              = float(age_params[1])
    
    for feat_type in feature_types:
        
        if feat_type not in {'age', 'general'}:
            # gather frequent feature indices from original fold
            with open(orig_feature_names_file_dict[feat_type], 'r') as f:
                feature_names = json.load(f)
            with open(orig_fold_feature_names_file_dict[feat_type], 'r') as f:
                orig_fold_feature_names = json.load(f)
            freq_feat_idxs = []
            current_idx = 0
            for feat_name in orig_fold_feature_names:
                while orig_fold_feature_names[current_idx] != feat_name:
                    current_idx += 1
                freq_feat_idxs.append(current_idx)
                current_idx += 1
            logger.info('Time to gather frequent ' + feat_type + ' feature indices from original fold: '
                        + str(time.time() - this_start_time) + ' seconds')
            
        for year_idx in range(num_years):
            year_X_dict  = {data_split: [] for data_split in data_splits}
            for month_idx in range(12*year_idx, 12*(year_idx+1)):
                this_start_time = time.time()
                month_windowed_features_hf5_file = windowed_features_fileheader_dict[feat_type] + str(month_idx) + '.hf5'
                if feat_type == 'age':
                    month_windowed_features = load_data_from_h5py(month_windowed_features_hf5_file)['age']
                    month_windowed_features = (month_windowed_features - age_mean)/age_std
                else:
                    month_windowed_features = load_sparse_matrix_from_h5py(month_windowed_features_hf5_file)
                    if feat_type != 'general':
                        month_windowed_features = csr_matrix(csc_matrix(month_windowed_features)[:,freq_feat_idxs])
                for data_split in data_splits:
                    year_X_dict[data_split].append(month_windowed_features[split_idxs_dict[data_split][month_idx]])
                logger.info('Time to load a month of ' + feat_type + ' data: ' 
                            + str(time.time() - this_start_time) + ' seconds')
                
            this_start_time = time.time()
            for data_split in data_splits:
                if feat_type == 'age':
                    save_data_to_h5py(features_file_dict[feat_type][data_split][year_idx],
                                      {'age': np.vstack(year_X_dict[data_split])[shuffle_idxs_dict[data_split][year_idx]]})
                else:
                    save_sparse_matrix_to_h5py(features_file_dict[feat_type][data_split][year_idx],
                                               vstack(year_X_dict[data_split], 
                                                      format='csr', 
                                                      dtype=np.int8)[shuffle_idxs_dict[data_split][year_idx]])
            logger.info('Time to stack, shuffle, and save a year of ' + feat_type + ' data: '
                        + str(time.time() - this_start_time) + ' seconds')
    logger.info('Time to run gather_fold_data_with_other_fold_features: ' + str(time.time() - start_time) + ' seconds')
    
def plot_cohort_size_and_outcome_freq(output_fileheader,
                                      plot_title,
                                      Y_all_years_dict,
                                      starting_year,
                                      logger,
                                      lab_outcome_id   = None,
                                      overwrite        = False,
                                      eligibility_time = None,
                                      debug_size       = None,
                                      fig              = None,
                                      ax               = None,
                                      cohort_name      = 'Cohort',
                                      linestyle        = '-'):
    '''
    Plot cohort size and outcome frequency summed across all data splits
    cohort size on left y-axis and outcome frequency on right y-axis
    If lab outcome is provided, then also plot cohort size with lab measured
    Save to json. Reads from json if exists unless overwrite specified. 
    If plot exists, will not replot unless overwrite specified.
    Plot will be added to existing figure and axes if specified.
    @param output_fileheader: str, start of path to output file, .json and .pdf will be appended
    @param plot_title: str, title of plots
    @param Y_all_years_dict: dict mapping str of data split to list of np arrays, outcomes for each year
    @param starting_year: int, starting year for labeling x-axis
    @param logger: logger, for INFO messages
    @param lab_outcome_id: str, comma-separated list of lab concept IDs if outcome is lab
    @param overwrite: bool, overwrites pdf and json if True
    @param eligibility_time: str, specifies time for eligibility criteria, only applied if plotting lab frequency
    @param debug_size: int, specifies limited cohort size for debugging, otherwise None, only applied if plotting lab frequency
    @param fig: matplotlib figure, add plot to this figure if specified (will be modified and returned)
    @param ax: matplotlib axes, add plot to this axis if specified (will be modified and returned), 
               top axis has cohort size, 
               bottom axis has outcome frequency
    @param cohort_name: str, start of legend entry
    @param linestyle: str, matplotlib line style
    @return: 1. matplotlib figure
             2. matplotlib axes
    '''
    if fig is not None:
        assert ax is not None
        assert len(ax) == 2
    plot_filename = output_fileheader + '.pdf'
    save_plot     = True
    if (not overwrite) and os.path.exists(plot_filename):
        save_plot = False
    json_filename = output_fileheader + '.json'
    num_years     = len(Y_all_years_dict[list(Y_all_years_dict.keys())[0]])
    if (not overwrite) and os.path.exists(json_filename):
        with open(json_filename, 'r') as f:
            cohort_stats   = json.load(f)
            cohort_sizes   = cohort_stats['Cohort size']
            outcome_counts = cohort_stats['Outcome count']
            outcome_freqs  = cohort_stats['Outcome frequency']
            if lab_outcome_id is not None:
                lab_freqs  = cohort_stats['Cohort with lab']
        logger.info('Cohort stats read from json')
    else:
        start_time     = time.time()
        cohort_sizes   = np.zeros(num_years)
        outcome_counts = np.zeros(num_years)
        for year_idx, data_split in product(range(num_years), Y_all_years_dict.keys()):
            cohort_sizes[year_idx]   += len(Y_all_years_dict[data_split][year_idx])
            outcome_counts[year_idx] += np.sum(Y_all_years_dict[data_split][year_idx])
        outcome_freqs  = np.divide(outcome_counts, cohort_sizes)
        cohort_stats   = {'Cohort size'      : cohort_sizes.tolist(),
                          'Outcome count'    : outcome_counts.tolist(),
                          'Outcome frequency': outcome_freqs.tolist()}

        if lab_outcome_id is not None:
            # gather how many patients in cohort have lab result each month from database
            this_start_time = time.time()
            assert eligibility_time is not None
            debug_suffix    = ''
            if debug_size is not None:
                debug_suffix = '_debug' + str(debug_size)
            with open(join(dirname(abspath(__file__)), 'sql/select_lab_frequency_over_time.sql'), 'r') as f:
                lab_freq_sql = f.read()
            lab_freq_sql     = lab_freq_sql.format(concept_id   = lab_outcome_id,
                                                   schema_name  = config.nonstationarity_schema_name,
                                                   window_name  = eligibility_time,
                                                   debug_suffix = debug_suffix)
            engine           = config.create_sqlalchemy_engine()
            with session_scope(engine) as session:
                lab_freq_result = session.execute(lab_freq_sql)
                session.commit()
            lab_freqs = np.zeros(num_years)
            month_idx = 0
            for row in lab_freq_result:
                lab_freqs[int(month_idx/12)] += int(row[1])
                month_idx += 1
            assert month_idx == 72

            cohort_stats['Cohort with lab'] = lab_freqs.tolist()
        with open(json_filename, 'w') as f:
            json.dump(cohort_stats, f)
        logger.info('Time to compute cohort stats: ' + str(time.time() - start_time) + ' seconds')
    
    start_time = time.time()
    plt.rcParams.update({'font.size': 14})
    if fig is None:
        plt.clf()
        fig, ax    = plt.subplots(nrows       = 2,
                                  ncols       = 1,
                                  figsize     = (8, 8),
                                  gridspec_kw = {'height_ratios': [3, 1]},
                                  sharex      = True)
    
    ax[0].plot(np.arange(starting_year, starting_year + num_years), 
               cohort_sizes,
               c         = 'black',
               label     = cohort_name + ' overall',
               linestyle = linestyle)
    if lab_outcome_id is not None:
        ax[0].plot(np.arange(starting_year, starting_year + num_years),
                   lab_freqs,
                   c         = 'blue',
                   label     = cohort_name + ' with lab',
                   linestyle = linestyle)
    ax[0].plot(np.arange(starting_year, starting_year + num_years), 
               outcome_counts,
               c         = 'red',
               label     = cohort_name + ' with outcome',
               linestyle = linestyle)
    ax[1].plot(np.arange(starting_year, starting_year + num_years),
               outcome_freqs,
               c         = 'red',
               label     = cohort_name + ' outcome frequency',
               linestyle = linestyle)
    if save_plot:
        ax[0].set_ylabel('Cohort size')
        ax0_max = max(ax[0].get_ylim()[1], max(cohort_sizes) * 1.05)
        ax[0].set_ylim([0, ax0_max])
        ax[0].set_title(plot_title)
        ax[0].legend()
        
        ax[1].set_ylabel('Outcome frequency')
        ax1_max = max(ax[1].get_ylim()[1], max(outcome_freqs) * 1.05)
        ax[1].set_ylim([0, ax1_max])
        ax[1].set_xlabel('Year')
        ax[1].set_xlim([starting_year, starting_year + num_years - 1])
        ax[1].set_xticks(range(starting_year, starting_year + num_years))
        
        ax[1].legend()
        fig.set_tight_layout(True)
        fig.savefig(plot_filename)
        logger.info('Saved cohort plot to ' + str(plot_filename))
    logger.info('Time to plot cohort stats: ' + str(time.time() - start_time) + ' seconds')
    return fig, ax