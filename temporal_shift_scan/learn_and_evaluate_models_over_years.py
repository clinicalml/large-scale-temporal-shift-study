import os
import sys
import json
import numpy as np
import time
import joblib

from os.path import dirname, abspath, join
sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from model_utils import (
    train_logreg, 
    save_logreg, 
    train_dectree, 
    save_dectree, 
    train_forest,
    save_forest,
    train_xgboost,
    save_xgboost,
    load_xgboost,
    eval_predictions, 
    compute_best_threshold,
    plot_precision_recall_curve
)

def learn_models_over_years(train_Xs, 
                            train_Ys, 
                            valid_Xs,
                            valid_Ys,
                            model_type,
                            feature_names,
                            output_file_header,
                            set_up_name,
                            logger,
                            year_idxs           = None,
                            feature_names_short = None,
                            overwrite           = False,
                            default_thresholds  = True):
    '''
    Learn a logistic regression, decision tree, random forest, or XGBoost classifier and a threshold for each year
    If a year has fewer than 10 validation samples with either outcome, no model is learned for that year
    Save models with joblib or xgboost
    Write coefficients to csv for logistic regressions. Save trees to text file
    Save thresholds to json
    @param train_Xs: list of csr matrices, training sample features at each time point
    @param train_Ys: list of np arrays, training sample outcomes at each time point
    @param valid_Xs: list of csr matrices, validation sample features at each time point
    @param valid_Ys: list of np arrays, validation sample outcomes at each time point
    @param model_type: str, logreg, dectree, forest, or xgboost
    @param feature_names: list of str, names of features
    @param output_file_header: str, start of file names
    @param set_up_name: str, name of set up for precision-recall curve plot title
    @param logger: logger, for INFO messages
    @param year_idxs: list of ints, indices of the time points for model file names
    @param feature_names_short: list of str, shortened names of features for visualizing decision tree
    @param overwrite: bool, whether to overwrite models and thresholds if they already exist
    @param default_thresholds: bool, use .5 for all thresholds if True, otherwise, compute best threshold with fpr criteria
    @return: 1. list of sklearn LogisticRegressions, DecisionTreeClassifiers, RandomForestClassifiers, or xgboost XGBClassifiers,
                one per year, None if insufficient samples to learn
             2. list of floats, thresholds for each year, None if insufficient samples to learn
             3. bool, whether models and thresholds were newly learned
    '''
    assert model_type in {'logreg', 'dectree', 'forest', 'xgboost'}
    if model_type == 'logreg':
        full_model_name = 'logistic_regression'
    elif model_type == 'dectree':
        full_model_name = 'decision_tree'
        assert len(feature_names_short) == len(feature_names)
    elif model_type == 'forest':
        full_model_name = 'random_forest'
    else:
        full_model_name = 'xgboost' 
    
    # check number of time points match
    num_years = len(train_Xs)
    assert num_years > 0
    assert len(train_Ys) == num_years
    assert len(valid_Xs) == num_years
    assert len(valid_Ys) == num_years
    if year_idxs is not None:
        assert len(year_idxs) == num_years
    else:
        year_idxs = range(num_years)
    
    # check number of features match at each time point
    assert np.all(np.array([train_Xs[t].shape[1] for t in range(len(train_Xs))]) == len(feature_names))
    assert np.all(np.array([valid_Xs[t].shape[1] for t in range(len(valid_Xs))]) == len(feature_names))
    logger.info(str(train_Xs[0].shape[1]) + ' features')
    
    # check number of samples matches between X and Y
    assert np.all(np.equal(np.array([train_Xs[t].shape[0] for t in range(len(train_Xs))]),
                           np.array([train_Ys[t].shape[0] for t in range(len(train_Ys))])))
    assert np.all(np.equal(np.array([valid_Xs[t].shape[0] for t in range(len(valid_Xs))]),
                           np.array([valid_Ys[t].shape[0] for t in range(len(valid_Ys))])))
        
    min_outcome_req     = 10
    models              = []
    thresholds_changed  = False
    loaded_thresholds   = False
    thresholds_filename = output_file_header + full_model_name + '_thresholds.json'
    if (not overwrite) and os.path.exists(thresholds_filename):
        with open(thresholds_filename, 'r') as f:
            thresholds = np.array(json.load(f))
        if (default_thresholds and np.all(np.logical_or(thresholds == .5, thresholds == -1))) \
            or ((not default_thresholds) and (not np.all(np.logical_or(thresholds == .5, thresholds == -1)))):
            loaded_thresholds = True
            logger.info('Loaded thresholds from ' + thresholds_filename)
        else:
            thresholds = -1 * np.ones(num_years)
    else:
        thresholds     = -1 * np.ones(num_years)
    
    for year_idx in range(num_years):
        start_time         = time.time()
        model_file_header  = output_file_header + full_model_name + '_time' + str(year_idxs[year_idx])
        if model_type == 'xgboost':
            model_filename = model_file_header + '.pkl'
        else:
            model_filename = model_file_header + '.joblib'
        
        num_valid_outcomes = np.sum(valid_Ys[year_idx])
        if num_valid_outcomes < min_outcome_req or num_valid_outcomes > len(valid_Ys[year_idx]) - min_outcome_req:
            if os.path.exists(model_filename):
                if model_type == 'xgboost':
                    os.rename(model_filename, model_file_header + '_old_model.pkl')
                    os.rename(model_file_header + '_params.json', model_file_header + '_old_model_params.json')
                else:
                    os.rename(model_filename, model_file_header + '_old_model.joblib')
            models.append(None)
            if thresholds[year_idx] != -1:
                thresholds[year_idx] = -1
                thresholds_changed   = True
            logger.info('Not enough validation samples with different outcomes to learn a model in year ' 
                        + str(year_idx))
            continue
        
        if (not overwrite) and os.path.exists(model_filename):
            if model_type == 'xgboost':
                model         = load_xgboost(model_file_header,
                                             logger)
            else:
                model         = joblib.load(model_filename)
            new_model_learned = False
            logger.info('Loaded model for year ' + str(year_idx) + ' from ' + model_filename)
        else:
            logger.info('Learning model for year ' + str(year_idx))
            if model_type == 'logreg':
                model = train_logreg(train_Xs[year_idx],
                                     train_Ys[year_idx],
                                     valid_Xs[year_idx],
                                     valid_Ys[year_idx],
                                     logger)
                save_logreg(model,
                            model_file_header,
                            feature_names,
                            logger)
            elif model_type == 'dectree':
                model = train_dectree(train_Xs[year_idx],
                                      train_Ys[year_idx],
                                      valid_Xs[year_idx],
                                      valid_Ys[year_idx],
                                      logger)
                save_dectree(model,
                             model_file_header,
                             feature_names,
                             feature_names_short,
                             train_Xs[year_idx],
                             train_Ys[year_idx],
                             valid_Xs[year_idx],
                             valid_Ys[year_idx],
                             logger)
            elif model_type == 'forest':
                model = train_forest(train_Xs[year_idx],
                                     train_Ys[year_idx],
                                     valid_Xs[year_idx],
                                     valid_Ys[year_idx],
                                     logger,
                                     n_estimators_options     = [10, 100, 1000],
                                     min_samples_leaf_options = [100, 1000, 10000])
                save_forest(model,
                            model_file_header,
                            logger)
            else:
                model = train_xgboost(train_Xs[year_idx],
                                      train_Ys[year_idx],
                                      valid_Xs[year_idx],
                                      valid_Ys[year_idx],
                                      logger,
                                      n_estimators_options = [5, 10, 50],
                                      max_depth_options    = [3,  6, 10])
                save_xgboost(model,
                             model_file_header,
                             logger)
            new_model_learned = True
        models.append(model)
        
        if (not loaded_thresholds) or new_model_learned:
            valid_preds         = model.predict_proba(valid_Xs[year_idx])[:,1]
            model_type_readable = {'logreg' : 'Logistic regression',
                                   'dectree': 'Decision tree',
                                   'forest' : 'Random forest',
                                   'xgboost': 'XGBoost classifier'}
            plot_precision_recall_curve(valid_Ys[year_idx],
                                        valid_preds,
                                        set_up_name + ' year ' + str(year_idx),
                                        model_type_readable[model_type],
                                        model_file_header + '_precision_recall_curve.pdf')
            if default_thresholds:
                year_threshold   = .5
            else:
                year_threshold   = compute_best_threshold(valid_Ys[year_idx],
                                                          valid_preds,
                                                          'fpr',
                                                          logger,
                                                          .2)
            thresholds[year_idx] = year_threshold
            thresholds_changed   = True
        logger.info('Learned model and threshold for year ' + str(year_idxs[year_idx]) + ' in '
                    + str(time.time() - start_time) + ' seconds')
    if thresholds_changed:
        with open(thresholds_filename, 'w') as f:
            json.dump(thresholds.tolist(), f)
        logger.info('Wrote thresholds to ' + thresholds_filename)
    return models, thresholds, thresholds_changed

def evaluate_models_on_future_years(models,
                                    thresholds,
                                    Xs,
                                    Ys,
                                    output_file_header,
                                    logger,
                                    overwrite = False,
                                    weights   = None):
    '''
    Evaluate models from each year on all years that come after
    Writes metrics to json file. If exists, reads from json
    If a year has fewer than 10 samples with either outcome, that year is not evaluated
    @param models: list of sklearn or xgboost models, one per year, None if insufficient samples to learn
    @param thresholds: list of floats, thresholds for each year, None if insufficient samples to learn
    @param Xs: list of csr matrices, sample features at each time point
    @param Ys: list of np arrays, sample outcomes at each time point
    @param output_file_header: str, start of file names
    @param logger: logger, for INFO messages
    @param overwrite: bool, whether to overwrite metrics file if it already exists
    @param weights: list of np arrays, sample weights at each time point, unit weights if None
    @return: 1. dict mapping int logistic regression year index
                to int evaluation year index
                to str metric name
                to float metric value,
                years without logistic regressions or enough samples to evaluate are omitted,
                metrics from evaluating each model on future years
             2. dict mapping str metric name 
                to int logistic regression year index
                to list of float metric values over evaluation years,
                years without logistic regressions are omitted, years without enough samples to evaluate have -1,
                metrics to plot
    '''
    json_filename = output_file_header + 'test_metrics.json'
    if (not overwrite) and os.path.exists(json_filename):
        with open(json_filename, 'r') as f:
            json_contents = json.load(f)
        metrics_dict    = json_contents['metrics_dict']
        metrics_dict    = {int(model_year): 
                           {int(eval_year): metrics_dict[model_year][eval_year]
                            for eval_year in metrics_dict[model_year]}
                           for model_year in metrics_dict}
        metrics_to_plot = json_contents['metrics_to_plot']
        metrics_to_plot = {metric_name: 
                           {int(model_year): metrics_to_plot[metric_name][model_year]
                            for model_year in metrics_to_plot[metric_name]}
                           for metric_name in metrics_to_plot}
        return metrics_dict, metrics_to_plot
    
    # check number of time points match
    num_years = len(models)
    assert num_years > 0
    assert len(thresholds) == num_years
    assert len(Xs)    == num_years
    assert len(Ys)    == num_years
    if weights is not None:
        assert len(weights) == num_years
    
    min_outcome_req = 10
    metrics_dict    = dict() # model year to eval year to dictionary of metrics
    metric_names    = ['auc', 'precision', 'recall', 'average_precision', 'f1_score', 'brier']
    metrics_to_plot = {metric_name: dict()
                       for metric_name in metric_names} # metric name to model year to metrics to plot
    for model_year_idx in range(num_years):
        if models[model_year_idx] is None:
            continue
        this_start_time              = time.time()
        year_model                   = models[model_year_idx]
        metrics_dict[model_year_idx] = dict()
        for metric_name in metric_names:
            metrics_to_plot[metric_name][model_year_idx] = []
        for eval_year_idx in range(model_year_idx, num_years):
            if np.sum(Ys[eval_year_idx]) < min_outcome_req \
                or np.sum(Ys[eval_year_idx]) > len(Ys[eval_year_idx]) - min_outcome_req:
                logger.info('Most samples in year ' + str(eval_year_idx) + ' have the same outcome')
                for metric_name in metric_names:
                    metrics_to_plot[metric_name][model_year_idx].append(-1)
                continue
            year_pred             = year_model.predict_proba(Xs[eval_year_idx])[:,1]
            logger.info('Evaluating model from ' + str(model_year_idx) + ' on ' + str(eval_year_idx))
            if weights is not None:
                eval_year_weights = weights[eval_year_idx]
            else:
                eval_year_weights = None
            year_metrics_dict     = eval_predictions(Ys[eval_year_idx],
                                                     year_pred,
                                                     logger,
                                                     thresholds[model_year_idx],
                                                     eval_year_weights)
            metrics_dict[model_year_idx][eval_year_idx] = year_metrics_dict
            for metric_name in metric_names:
                metrics_to_plot[metric_name][model_year_idx].append(year_metrics_dict[metric_name])
        logger.info('Time to evaluate model from ' + str(model_year_idx) + ': ' 
                    + str(time.time() - this_start_time) + ' seconds')

    json_contents = {'metrics_dict'   : metrics_dict,
                     'metrics_to_plot': metrics_to_plot}
    with open(json_filename, 'w') as f:
        json.dump(json_contents, f)
    return metrics_dict, metrics_to_plot