import numpy as np
import multiprocessing as mp
import time
import math
import joblib
import json
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from itertools import product
from scipy.sparse import csc_matrix, hstack

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, 
    brier_score_loss,
    average_precision_score, 
    f1_score,
    recall_score,
    precision_score,
    precision_recall_curve,
    PrecisionRecallDisplay
)
from xgboost import XGBClassifier
import statsmodels.api as sm

from logging_utils import log_or_print

def train_single_logreg(C,
                        X_train,
                        y_train,
                        penalty,
                        solver,
                        sample_weight_train = None):
    '''
    Train a single logistic regression.
    This method is only used for parallelization in train_logreg.
    @param C: float, regularization
    @param X_train: csr matrix or np array, # samples x # features
    @param y_train: csr matrix or np array, # samples
    @param penalty: str, l1 or l2
    @param solver: str, liblinear or lbfgs
    @param sample_weight_train: np array of floats, sample weights for training logistic regression, None for unit weights
    @return: 1. sklearn LogisticRegression
             2. float, number of seconds to fit model
    '''
    assert penalty in {'l1', 'l2'}
    assert solver in {'liblinear', 'lbfgs'}
    assert X_train.shape[0] == len(y_train)
    if sample_weight_train is not None:
        assert len(sample_weight_train) == len(y_train)
    this_start_time = time.time()
    logreg = LogisticRegression(penalty      = penalty,
                                C            = C, 
                                solver       = solver,
                                multi_class  = 'auto',
                                class_weight = 'balanced',
                                random_state = 0, 
                                max_iter     = 1000)

    logreg.fit(X_train, 
               y_train,
               sample_weight = sample_weight_train)
    return logreg, time.time() - this_start_time

def train_logreg(X_train, 
                 y_train, 
                 X_valid, 
                 y_valid,
                 logger              = None,
                 Cs                  = [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
                 penalty             = 'l2',
                 sample_weight_train = None,
                 sample_weight_valid = None):
    '''
    Train a logistic regression with tuning for L1 or L2 regularization hyperparameter.
    @param X_train: np or sparse array of training sample features
    @param y_train: binary outcomes for training samples
    @param X_valid: np or sparse array of validation sample features
    @param y_valid: binary outcomes for validation samples
    @param logger: logger, for INFO messages, if None will print to stdout
    @param Cs: list of floats, L1 or L2 regularization hyperparameters
    @param penalty: str, l1 or l2
    @param sample_weight_train: np array of floats, sample weights for training model, None for unit weights
    @param sample_weight_valid: np array of floats, sample weights for evaluating model, None for unit weights
    @return: sklearn LogisticRegression
    '''
    assert X_train.shape[0] == len(y_train)
    assert X_train.shape[1] == X_valid.shape[1]
    assert X_valid.shape[0] == len(y_valid)
    assert len(Cs) > 0
    assert penalty in {'l1', 'l2'}
    if sample_weight_train is not None:
        assert len(sample_weight_train) == len(y_train)
    if sample_weight_valid is not None:
        assert len(sample_weight_valid) == len(y_valid)
    if penalty == 'l2':
        solver = 'lbfgs'
    else:
        solver = 'liblinear'
    
    this_start_time = time.time()
    with mp.get_context('spawn').Pool(processes=min(len(Cs), mp.cpu_count())) as pool:
        logregs, times = zip(*pool.map(partial(train_single_logreg, 
                                               X_train             = X_train,
                                               y_train             = y_train,
                                               penalty             = penalty,
                                               solver              = solver,
                                               sample_weight_train = sample_weight_train),
                                       Cs))
    for C_idx in range(len(Cs)):
        log_or_print('Time to train logreg C=' + str(Cs[C_idx]) + ': ' + str(times[C_idx]) + ' seconds',
                     logger)
    log_or_print('Time to train all logregs: ' + str(time.time() - this_start_time) + ' seconds',
                 logger)
    
    best_valid_auc      = -1
    for C_idx in range(len(Cs)):
        this_start_time = time.time()
        C               = Cs[C_idx]
        logreg          = logregs[C_idx]
        valid_pred      = logreg.predict_proba(X_valid)[:, 1]
        valid_auc       = roc_auc_score(y_valid, 
                                        valid_pred,
                                        sample_weight=sample_weight_valid)
        log_or_print('Valid AUC for C=' + str(C) + ': {0:.4f}'.format(valid_auc),
                     logger)
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_logreg    = logreg
            best_C         = C
        log_or_print('Time to evaluate logreg C=' + str(C) + ': ' + str(time.time() - this_start_time) + ' seconds',
                     logger)

    log_or_print('Best C: ' + str(best_C),
                 logger)
    log_or_print('Best valid AUC: {0:.4f}'.format(best_valid_auc),
                 logger)
    return best_logreg

def save_logreg(logreg,
                file_header,
                feature_names,
                logger = None):
    '''
    Save logistic regression with joblib and save coefficients to text file
    @param logreg: sklearn LogisticRegression
    @param file_header: str, path to save files, .joblib and _coefficients.csv will be appended
    @param feature_names: list of str, corresponding to coefficients of logreg
    @param logger: logger, for INFO messages, if None will print to stdout
    @return: None
    '''
    assert len(logreg.coef_[0]) == len(feature_names)
    this_start_time = time.time()
    logreg_file     = file_header + '.joblib'
    joblib.dump(logreg,
                logreg_file)
    
    coefficients_file = file_header + '_coefficients.csv'
    coefficients_df   = pd.DataFrame({'Feature'    : feature_names, 
                                      'Coefficient': logreg.coef_[0]})
    coefficients_df   = coefficients_df.sort_values(by        = 'Coefficient',
                                                    ascending = False)
    coefficients_df.to_csv(coefficients_file,
                           index = False)
    log_or_print('Time to save logistic regression: ' + str(time.time() - this_start_time) + ' seconds',
                 logger)

def train_statsmodels_logreg(X_train,
                             y_train,
                             feature_names,
                             outcome_name,
                             logger = None):
    '''
    Train an un-regularized logistic regression using statsmodels
    @param X_train: np or sparse array of training sample features
    @param y_train: binary outcomes for training samples
    @param feature_names: list of str, corresponding to columns in feature matrix
    @param outcome_name: str, will be used for outcome name in statsmodels
    @param logger: logger, for INFO messages, if None will print to stdout
    @return: statsmodels LogitResults
    '''
    assert X_train.shape[0] == len(y_train)
    assert X_train.shape[1] == len(feature_names)
    assert np.all(np.logical_or(y_train == 0, y_train == 1))
    
    start_time  = time.time()
    X_train_const = hstack((X_train,
                            np.ones((X_train.shape[0], 1))),
                           format = 'csr')
    y_train_df  = pd.DataFrame(data    = {outcome_name: y_train},
                               columns = [outcome_name])
    log_or_print('Time to create dataframes for statsmodels: ' + str(time.time() - start_time) + ' seconds',
                 logger)
    
    start_time  = time.time()
    logreg      = sm.Logit(y_train_df,
                           X_train_const.toarray())
    logreg.exog_names[:]  = feature_names + ['constant']
    results     = logreg.fit(method  = 'lbfgs',
                             maxiter = 100,
                             disp    = 0)
    log_or_print('Time to fit statsmodels logistic regression: ' + str(time.time() - start_time) + ' seconds',
                 logger)
    
    return results
    
def save_statsmodels_logreg(logreg,
                            file_header,
                            logger = None):
    '''
    Save statsmodel logistic regression to a pickle file
    Save coefficients with confidence intervals to a csv
    @param logreg: statsmodels LogitResults
    @param file_header: str, path to save model, .pkl and _coefficients.csv will be appended
    @param logger: logger, for INFO messages
    @return: pandas DataFrame containing statsmodels summary
    '''
    start_time = time.time()
    logreg_summary = logreg.summary()
    summary_as_html = logreg_summary.tables[1].as_html()
    summary_df      = pd.read_html(summary_as_html, header=0, index_col=0)[0]
    summary_df.reset_index(inplace = True)
    summary_df.rename(columns = {'index': 'Feature'}, 
                      inplace = True)
    summary_df.to_csv(file_header + '_coefficients.csv',
                      index = False)
    
    # reduce file size by removing data, model can still make predictions when reloaded
    logreg.remove_data()
    logreg.save(file_header + '.pkl')
    
    log_or_print('Time to save statsmodels logistic regression and summary: ' + str(time.time() - start_time) + ' seconds',
                 logger)
    
    return summary_df

def get_predictions_from_statsmodels_logreg(logreg,
                                            X):
    '''
    Get predictions from statsmodels logistic regression
    @param logreg: statsmodels LogitResults
    @param X: csr matrix, covariates, # samples x # features
    @return: np array, predicted probabilities
    '''
    X_const = hstack((X,
                      np.ones((X.shape[0], 1))),
                     format = 'csr')
    return logreg.predict(X_const.toarray())
    
def compute_false_positives_and_negatives(y_true,
                                          y_pred_thresholded,
                                          logger = None):
    '''
    Log or print number of false positives and negatives
    @param y_true: np array, binary labels
    @param y_pred_thresholded: np array, binary predictions
    @param logger: logger, for INFO messages, if None will print to stdout
    @return: None
    '''
    assert len(y_true) == len(y_pred_thresholded)
    assert np.all(np.logical_or(y_true == 0, y_true == 1))
    assert np.all(np.logical_or(y_pred_thresholded == 0, y_pred_thresholded == 1))
    
    num_pos_outcomes            = np.sum(y_true)
    if num_pos_outcomes == 0:
        log_or_print('No samples with outcome = 1',
                     logger)
    else:
        num_pos_outcomes_missed     = np.sum(np.where(np.logical_and(y_true == 1, y_pred_thresholded == 0), 1, 0))
        percent_pos_outcomes_missed = 100 * num_pos_outcomes_missed / float(num_pos_outcomes)
        log_or_print(str(int(num_pos_outcomes_missed)) + ' of ' + str(int(num_pos_outcomes)) 
                     + ' samples (' + str(percent_pos_outcomes_missed) + '%) with outcome = 1 are missed',
                     logger)
    
    num_pred_pos      = np.sum(y_pred_thresholded)
    if num_pred_pos == 0:
        log_or_print('No samples with predicted outcome = 1',
                     logger)
    else:
        num_false_pos     = np.sum(np.where(np.logical_and(y_true == 0, y_pred_thresholded == 1), 1, 0))
        percent_false_pos = 100 * num_false_pos / float(num_pred_pos)
        log_or_print(str(int(num_false_pos)) + ' of ' + str(int(num_pred_pos)) 
                     + ' samples (' + str(percent_false_pos) + '%) with predicted outcome = 1 are false positives',
                     logger)

def eval_predictions(y_true, 
                     y_pred,
                     logger        = None,
                     threshold     = .5,
                     sample_weight = None):
    '''
    Evaluate auc, f1_score, average_precision, precision, recall, brier
    Compute standard deviations for auc and brier
    Compute upper and lower bounds for bootstrap percentile confidence interval for auc
    Log false positive and false negative rates
    @param y_true: np array, binary labels
    @param y_pred: np array, predictions between 0 and 1, same length
    @param logger: logger, for INFO messages, if None will print to stdout
    @param threshold: float, determines which probabilities are mapped to 0 or 1 for precision, recall, f1_score
    @param sample_weight: np array of floats, sample weights for evaluating metrics, None for unit weights
    @return: dictionary mapping name of metric to float value 
    '''
    assert len(y_true) == len(y_pred)
    assert np.all(np.logical_or( y_true == 0, y_true == 1))
    assert np.all(np.logical_and(y_pred >= 0, y_pred <= 1))
    assert threshold >= 0 and threshold <= 1
    if sample_weight is not None:
        assert len(sample_weight) == len(y_true)
        assert np.all(sample_weight >= 0)
    
    y_pred_thresholded = np.where(y_pred >= threshold, 1, 0)
    
    metrics = dict()
    metrics['f1_score']             = f1_score(y_true,
                                               y_pred_thresholded,
                                               sample_weight = sample_weight)
    metrics['precision']            = precision_score(y_true,
                                                      y_pred_thresholded,
                                                      sample_weight = sample_weight)
    metrics['recall']               = recall_score(y_true,
                                                   y_pred_thresholded,
                                                   sample_weight = sample_weight)
    metrics['average_precision']    = average_precision_score(y_true, 
                                                              y_pred,
                                                              sample_weight = sample_weight)
    
    metrics['auc']                  = roc_auc_score(y_true, 
                                                    y_pred,
                                                    sample_weight = sample_weight)
    
    metrics['brier']                = brier_score_loss(y_true,
                                                       y_pred,
                                                       sample_weight = sample_weight)
    
    for metric_name in metrics.keys():
        log_or_print(metric_name + ': {0:.4f}'.format(metrics[metric_name]),
                     logger)
    
    compute_false_positives_and_negatives(y_true,
                                          y_pred_thresholded,
                                          logger)
    return metrics

def plot_precision_recall_curve(y_true,
                                y_pred,
                                set_up_name,
                                model_name,
                                output_file_name,
                                sample_weight = None):
    '''
    Plot precision-recall curve for model
    @param y_true: np array, binary labels
    @param y_pred: np array, predictions between 0 and 1, same length
    @param set_up_name: str, name of set up for plot title
    @param model_name: str, name of model class for legend
    @param output_file_name: str, path to save plot
    @param sample_weight: np array of floats, sample weights for evaluating metrics, None for unit weights
    @return: None
    '''
    assert len(y_true) == len(y_pred)
    assert np.all(np.logical_or( y_true == 0, y_true == 1))
    assert np.all(np.logical_and(y_pred >= 0, y_pred <= 1))
    if sample_weight is not None:
        assert len(sample_weight) == len(y_true)
        assert np.all(sample_weight >= 0)
    
    plt.clf()
    display = PrecisionRecallDisplay.from_predictions(y_true, 
                                                      y_pred,
                                                      name          = model_name,
                                                      sample_weight = sample_weight)
    display.ax_.set_title(set_up_name)
    display.ax_.set_xlabel('Recall')
    display.ax_.set_ylabel('Precision')
    display.ax_.set_ylim([0,1])
    display.ax_.legend(loc = 'upper right')
    display.figure_.savefig(output_file_name)

def compute_best_threshold(y_true,
                           y_pred,
                           threshold_metric,
                           logger               = None,
                           threshold_metric_val = None,
                           sample_weight        = None):
    '''
    Compute threshold with one of the metrics
    @param y_true: np array, binary labels
    @param y_pred: np array, predictions between 0 and 1, same length
    @param threshold_metric: str, options: 1. f1 : smallest threshold with highest f1 score
                                           2. fpr: smallest threshold that achieves at most specified false positive rate
                                           3. fnr: largest  threshold that achieves at most specified false negative rate
    @param logger: logger, for INFO messages, if None will print to stdout
    @param threshold_metric_val: float, specify false positive / negative rate for threshold, between 0 and 1
    @param sample_weight: np array of floats, sample weights for evaluating metrics, None for unit weights
    @return: float, threshold
    '''
    assert len(y_true) == len(y_pred)
    assert np.all(np.logical_or( y_true == 0, y_true == 1))
    assert np.all(np.logical_and(y_pred >= 0, y_pred <= 1))
    assert threshold_metric in {'f1', 'fpr', 'fnr'}
    if  threshold_metric in {'fpr', 'fnr'}:
        assert threshold_metric_val >= 0
        assert threshold_metric_val <= 1
    if sample_weight is not None:
        assert len(sample_weight) == len(y_true)
        assert np.all(sample_weight >= 0)
    
    precisions, recalls, thresholds = precision_recall_curve(y_true,
                                                             y_pred,
                                                             sample_weight = sample_weight)
    best_threshold                  = None
    best_precision                  = None
    best_recall                     = None
    if threshold_metric == 'f1':
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            f1_scores               = np.where(precisions + recalls == 0,
                                               -1,
                                               (2 * precisions * recalls) / (precisions + recalls))
        best_threshold_idx          = np.argmax(f1_scores)
    elif threshold_metric == 'fpr':
        fpr_below20_threshold_idxs  = np.argwhere(precisions >= (1 - threshold_metric_val))
        if len(fpr_below20_threshold_idxs) == 0:
            best_threshold_idx      = len(thresholds)
        else:
            best_threshold_idx      = fpr_below20_threshold_idxs[0][0]
    else:
        fnr_below20_threshold_idxs  = np.argwhere(recalls    >= (1 - threshold_metric_val))
        if len(fnr_below20_threshold_idxs) == 0:
            best_threshold          = 0
            best_precision          = np.mean(y_true)
            best_recall             = 1
        else:
            best_threshold_idx      = fnr_below20_threshold_idxs[-1][0]
    if best_threshold is None:
        if best_threshold_idx == len(thresholds):
            best_threshold          = 1
        else:
            best_threshold          = thresholds[best_threshold_idx]
    if best_precision is None:
        best_precision              = precisions[best_threshold_idx]
    if best_recall is None:
        best_recall                 = recalls[best_threshold_idx]
    if best_precision + best_recall == 0:
        best_f1_score               = 0
    else:
        best_f1_score               = (2 * best_precision * best_recall) / (best_precision + best_recall)
    
    log_or_print('Threshold: ' + str(best_threshold),
                 logger)
    log_or_print('Precision: ' + str(best_precision),
                 logger)
    log_or_print('Recall: '    + str(best_recall),
                 logger)
    log_or_print('F1 score: '  + str(best_f1_score),
                 logger)
    y_pred_thresholded              = np.where(y_pred >= best_threshold, 1, 0)
    compute_false_positives_and_negatives(y_true,
                                          y_pred_thresholded,
                                          logger)
    return best_threshold

def train_single_dectree(min_samples_leaf,
                         X_train,
                         y_train,
                         sample_weight_train = None):
    '''
    Train a single decision tree.
    This method is only used for parallelization in train_dectree.
    @param min_samples_leaf: int, hyperparameter
    @param X_train: csr matrix or np array, # samples x # features
    @param y_train: csr matrix or np array, # samples
    @param sample_weight_train: np array of floats, sample weights for training logistic regression, None for unit weights
    @return: 1. sklearn DecisionTreeClassifier
             2. float, number of seconds to fit model
    '''
    assert X_train.shape[0] == len(y_train)
    if sample_weight_train is not None:
        assert len(sample_weight_train) == len(y_train)
    assert min_samples_leaf > 0
    
    this_start_time = time.time()
    dectree = DecisionTreeClassifier(min_samples_leaf = min_samples_leaf,
                                     class_weight     = 'balanced',
                                     random_state     = 0)
    dectree.fit(X_train, 
                y_train,
                sample_weight = sample_weight_train)
    return dectree, time.time() - this_start_time

def train_dectree(X_train, 
                  y_train, 
                  X_valid, 
                  y_valid,
                  logger                   = None,
                  min_samples_leaf_options = [10, 25, 100],
                  sample_weight_train      = None,
                  sample_weight_valid      = None):
    '''
    Train a decision tree with tuning for min samples per leaf hyperparameter.
    @param X_train: np or sparse array of training sample features
    @param y_train: binary outcomes for training samples
    @param X_valid: np or sparse array of validation sample features
    @param y_valid: binary outcomes for validation samples
    @param logger: logger, for INFO messages, if None will print to stdout
    @param min_samples_leaf_options: list of ints, options for hyperparameter tuning
    @param sample_weight_train: np array of floats, sample weights for training model, None for unit weights
    @param sample_weight_valid: np array of floats, sample weights for evaluating model, None for unit weights
    @return: sklearn DecisionTreeClassifier
    '''
    assert X_train.shape[0] == len(y_train)
    assert X_train.shape[1] == X_valid.shape[1]
    assert X_valid.shape[0] == len(y_valid)
    assert len(min_samples_leaf_options) > 0
    assert np.all(np.array(min_samples_leaf_options) > 0)
    if sample_weight_train is not None:
        assert len(sample_weight_train) == len(y_train)
    if sample_weight_valid is not None:
        assert len(sample_weight_valid) == len(y_valid)
    
    this_start_time = time.time()
    with mp.get_context('spawn').Pool(processes=min(len(min_samples_leaf_options), mp.cpu_count())) as pool:
        dectrees, times = zip(*pool.map(partial(train_single_dectree, 
                                                X_train             = X_train,
                                                y_train             = y_train,
                                                sample_weight_train = sample_weight_train),
                                        min_samples_leaf_options))
    for min_samples_leaf_idx in range(len(min_samples_leaf_options)):
        log_or_print('Time to train dectree min_samples_leaf=' + str(min_samples_leaf_options[min_samples_leaf_idx]) + ': ' 
                     + str(times[min_samples_leaf_idx]) + ' seconds',
                     logger)
    log_or_print('Time to train all dectrees: ' + str(time.time() - this_start_time) + ' seconds',
                 logger)
    
    best_valid_auc       = -1
    for min_samples_leaf_idx in range(len(min_samples_leaf_options)):
        this_start_time  = time.time()
        min_samples_leaf = min_samples_leaf_options[min_samples_leaf_idx]
        dectree          = dectrees[min_samples_leaf_idx]
        valid_pred       = dectree.predict_proba(X_valid)[:, 1]
        valid_auc        = roc_auc_score(y_valid, 
                                         valid_pred,
                                         sample_weight = sample_weight_valid)
        log_or_print('Valid AUC for min_samples_leaf=' + str(min_samples_leaf) + ': {0:.4f}'.format(valid_auc),
                     logger)
        if valid_auc > best_valid_auc:
            best_valid_auc        = valid_auc
            best_dectree          = dectree
            best_min_samples_leaf = min_samples_leaf
        log_or_print('Time to evaluate dectree min_samples_leaf=' + str(min_samples_leaf) + ': ' 
                     + str(time.time() - this_start_time) + ' seconds',
                     logger)

    log_or_print('Best min_samples_leaf: ' + str(best_min_samples_leaf),
                 logger)
    log_or_print('Best valid AUC: {0:.4f}'.format(best_valid_auc),
                 logger)
    return best_dectree

def save_dectree(dectree,
                 file_header,
                 feature_names,
                 feature_names_short,
                 X_train,
                 y_train,
                 X_valid, 
                 y_valid,
                 logger = None):
    '''
    Save decision tree with joblib and save diagram to file
    @param dectree: sklearn DecisionTreeClassifier
    @param file_header: str, path to save files, .joblib, .pdf, and .txt will be appended
    @param feature_names: list of str, corresponding to features in decision tree, longer used for printing tree
    @param feature_names_short: list of str, corresponding to features in decision tree, shorter used for plotting tree
    @param X_train: np or sparse array of training sample features
    @param y_train: binary outcomes for training samples
    @param X_valid: np or sparse array of validation sample features
    @param y_valid: binary outcomes for validation samples
    @param logger: logger, for INFO messages, if None will print to stdout
    @return: None
    '''
    assert dectree.n_features_in_ == len(feature_names)
    assert dectree.n_features_in_ == len(feature_names_short)
    this_start_time = time.time()
    dectree_file    = file_header + '.joblib'
    joblib.dump(dectree,
                dectree_file)
    
    plt.clf()
    plt.figure(figsize=(24,12))
    plot_tree(dectree,
              max_depth     = 3,
              feature_names = feature_names_short,
              fontsize      = 12)
    plt.savefig(file_header + '.pdf',
                dpi = 100)
    
    train_pred_leaves = dectree.apply(X_train)
    valid_pred_leaves = dectree.apply(X_valid)
    def describe_node(i,
                      depth):
        '''
        Create a string describing tree from node i downwards
        @param i: int, node index
        @param depth: int, depth of node in tree, for adding tabs before description
        @return: str, description
        '''
        assert i >= 0 and i < dectree.tree_.node_count
        assert depth >= 0
        if dectree.tree_.children_left[i] == dectree.tree_.children_right[i]:
            train_idxs_at_leaf   = np.argwhere(train_pred_leaves == i)
            num_train_at_leaf    = len(train_idxs_at_leaf)
            num_train_y1_at_leaf = np.sum(y_train[train_idxs_at_leaf])
            valid_idxs_at_leaf   = np.argwhere(valid_pred_leaves == i)
            num_valid_at_leaf    = len(valid_idxs_at_leaf)
            num_valid_y1_at_leaf = np.sum(y_valid[valid_idxs_at_leaf])
            output_str = 'train: ' + str(num_train_y1_at_leaf) + '/' + str(num_train_at_leaf) \
                       + ', valid: ' + str(num_valid_y1_at_leaf) + '/' + str(num_valid_at_leaf) + '\n'
            return output_str
        
        output_str = feature_names[dectree.tree_.feature[i]] + ' <= ' + str(dectree.tree_.threshold[i]) + '?\n' \
                   + (depth + 1) * '\t' + 'yes: ' + describe_node(dectree.tree_.children_left[i],
                                                                  depth + 1) \
                   + (depth + 1) * '\t' + 'no: '  + describe_node(dectree.tree_.children_right[i],
                                                                  depth + 1)
        return output_str
    tree_description = describe_node(0, 0)
    with open(file_header + '.txt', 'w') as f:
        f.write(tree_description)
    log_or_print('Time to save, plot, and describe decision tree: ' + str(time.time() - this_start_time) + ' seconds',
                 logger)
    
def train_single_forest(hyperparams,
                        X_train,
                        y_train,
                        sample_weight_train = None):
    '''
    Train a single random forest classifier.
    This method is only used for parallelization in train_forest
    @param hyperparams: dict, maps str name of hyperparameter to int value, contains n_estimators and min_samples_leaf
    @param X_train: csr matrix or np array, # samples x # features
    @param y_train: csr matrix or np array, # samples
    @param sample_weight_train: np array of floats, sample weights for training logistic regression, None for unit weights
    @return: 1. sklearn RandomForestClassifier
             2. float, number of seconds to fit model
    '''
    assert X_train.shape[0] == len(y_train)
    if sample_weight_train is not None:
        assert len(sample_weight_train) == len(y_train)
    assert 'n_estimators' in hyperparams
    assert hyperparams['n_estimators'] > 0
    assert 'min_samples_leaf' in hyperparams
    assert hyperparams['min_samples_leaf'] > 0
        
    this_start_time = time.time()
    forest = RandomForestClassifier(n_estimators     = hyperparams['n_estimators'],
                                    min_samples_leaf = hyperparams['min_samples_leaf'],
                                    class_weight     = 'balanced',
                                    random_state     = 0)
    forest.fit(X_train, 
                y_train,
                sample_weight = sample_weight_train)
    return forest, time.time() - this_start_time

def train_forest(X_train, 
                 y_train, 
                 X_valid, 
                 y_valid,
                 logger                   = None,
                 n_estimators_options     = [10, 25, 100],
                 min_samples_leaf_options = [10, 25, 100],
                 sample_weight_train      = None,
                 sample_weight_valid      = None):
    '''
    Train a random forest with tuning for number of estimators and min samples per leaf hyperparameters.
    @param X_train: np or sparse array of training sample features
    @param y_train: binary outcomes for training samples
    @param X_valid: np or sparse array of validation sample features
    @param y_valid: binary outcomes for validation samples
    @param logger: logger, for INFO messages, if None will print to stdout
    @param n_estimators_options: list of ints, options for hyperparameter tuning
    @param min_samples_leaf_options: list of ints, options for hyperparameter tuning
    @param sample_weight_train: np array of floats, sample weights for training model, None for unit weights
    @param sample_weight_valid: np array of floats, sample weights for evaluating model, None for unit weights
    @return: sklearn RandomForestClassifier
    '''
    assert X_train.shape[0] == len(y_train)
    assert X_train.shape[1] == X_valid.shape[1]
    assert X_valid.shape[0] == len(y_valid)
    assert len(n_estimators_options) > 0
    assert np.all(np.array(n_estimators_options) > 0)
    assert len(min_samples_leaf_options) > 0
    assert np.all(np.array(min_samples_leaf_options) > 0)
    if sample_weight_train is not None:
        assert len(sample_weight_train) == len(y_train)
    if sample_weight_valid is not None:
        assert len(sample_weight_valid) == len(y_valid)
    
    this_start_time = time.time()
    hyperparam_dicts = [{'n_estimators'    : n_estimators,
                         'min_samples_leaf': min_samples_leaf}
                        for n_estimators, min_samples_leaf 
                        in product(n_estimators_options, min_samples_leaf_options)]
    with mp.get_context('spawn').Pool(processes=min(len(hyperparam_dicts), mp.cpu_count())) as pool:
        forests, times = zip(*pool.map(partial(train_single_forest, 
                                                X_train             = X_train,
                                                y_train             = y_train,
                                                sample_weight_train = sample_weight_train),
                                        hyperparam_dicts))
    for hyperparam_dict_idx in range(len(hyperparam_dicts)):
        log_or_print('Time to train random forest n_estimators=' + str(hyperparam_dicts[hyperparam_dict_idx]['n_estimators']) 
                     + ', min_samples_leaf=' + str(hyperparam_dicts[hyperparam_dict_idx]['min_samples_leaf']) + ': ' 
                     + str(times[hyperparam_dict_idx]) + ' seconds',
                     logger)
    log_or_print('Time to train all random forests: ' + str(time.time() - this_start_time) + ' seconds',
                 logger)
    
    best_valid_auc       = -1
    for hyperparam_dict_idx in range(len(hyperparam_dicts)):
        this_start_time  = time.time()
        n_estimators     = hyperparam_dicts[hyperparam_dict_idx]['n_estimators']
        min_samples_leaf = hyperparam_dicts[hyperparam_dict_idx]['min_samples_leaf']
        forest           = forests[hyperparam_dict_idx]
        valid_pred       = forest.predict_proba(X_valid)[:, 1]
        valid_auc        = roc_auc_score(y_valid, 
                                         valid_pred,
                                         sample_weight = sample_weight_valid)
        log_or_print('Valid AUC for n_estimators=' + str(n_estimators) + ', min_samples_leaf=' + str(min_samples_leaf) 
                     + ': {0:.4f}'.format(valid_auc),
                     logger)
        if valid_auc > best_valid_auc:
            best_valid_auc        = valid_auc
            best_forest           = forest
            best_n_estimators     = n_estimators
            best_min_samples_leaf = min_samples_leaf
        log_or_print('Time to evaluate random forest with n_estimators=' + str(n_estimators) 
                     + ', min_samples_leaf=' + str(min_samples_leaf) + ': ' + str(time.time() - this_start_time) + ' seconds',
                     logger)

    log_or_print('Best n_estimators: ' + str(best_n_estimators),
                 logger)
    log_or_print('Best min_samples_leaf: ' + str(best_min_samples_leaf),
                 logger)
    log_or_print('Best valid AUC: {0:.4f}'.format(best_valid_auc),
                 logger)
    return best_forest

def save_forest(forest,
                file_header,
                logger = None):
    '''
    Save random forest with joblib
    @param forest: sklearn RandomForestClassifier
    @param file_header: str, path to save file, .joblib will be appended
    @param logger: logger, for INFO messages, if None will print to stdout
    @return: None
    '''
    this_start_time = time.time()
    forest_file    = file_header + '.joblib'
    joblib.dump(forest,
                forest_file)
    log_or_print('Time to save random forest: ' + str(time.time() - this_start_time) + ' seconds',
                 logger)
    
def train_single_xgboost(hyperparams,
                         X_train,
                         y_train,
                         sample_weight_train = None):
    '''
    Train a single random XGBoost classifier.
    This method is only used for parallelization in train_xgboost
    @param hyperparams: dict, maps str name of hyperparameter to int value, contains n_estimators and max_depth
    @param X_train: csr matrix or np array, # samples x # features
    @param y_train: csr matrix or np array, # samples
    @param sample_weight_train: np array of floats, sample weights for training logistic regression, None for unit weights
    @return: 1. xgboost XGBClassifier
             2. float, number of seconds to fit model
    '''
    assert X_train.shape[0] == len(y_train)
    if sample_weight_train is not None:
        assert len(sample_weight_train) == len(y_train)
    assert 'n_estimators' in hyperparams
    assert hyperparams['n_estimators'] > 0
    assert 'max_depth' in hyperparams
    assert hyperparams['max_depth'] > 0
        
    this_start_time  = time.time()
    num_y1_samples   = y_train.sum()
    num_y0_samples   = len(y_train) - num_y1_samples
    scale_pos_weight = num_y0_samples / float(num_y1_samples)
    model = XGBClassifier(n_estimators      = hyperparams['n_estimators'],
                          max_depth         = hyperparams['max_depth'],
                          scale_pos_weight  = scale_pos_weight,
                          random_state      = 0,
                          use_label_encoder = False,
                          eval_metric       = 'logloss')
    model.fit(X_train, 
              y_train,
              sample_weight = sample_weight_train)
    return model, time.time() - this_start_time
    
def train_xgboost(X_train,
                  y_train,
                  X_valid,
                  y_valid,
                  logger               = None,
                  n_estimators_options = [10, 25, 100],
                  max_depth_options    = [3,  6,  10],
                  sample_weight_train      = None,
                  sample_weight_valid      = None):
    '''
    Train a XGBoost classifier with tuning for number of estimators and maximum depth hyperparameters.
    @param X_train: np or sparse array of training sample features
    @param y_train: binary outcomes for training samples
    @param X_valid: np or sparse array of validation sample features
    @param y_valid: binary outcomes for validation samples
    @param logger: logger, for INFO messages, if None will print to stdout
    @param n_estimators_options: list of ints, options for hyperparameter tuning
    @param max_depth_options: list of ints, options for hyperparameter tuning
    @param sample_weight_train: np array of floats, sample weights for training model, None for unit weights
    @param sample_weight_valid: np array of floats, sample weights for evaluating model, None for unit weights
    @return: xgboost XGBClassifier
    '''
    assert X_train.shape[0] == len(y_train)
    assert X_train.shape[1] == X_valid.shape[1]
    assert X_valid.shape[0] == len(y_valid)
    assert len(n_estimators_options) > 0
    assert np.all(np.array(n_estimators_options) > 0)
    assert len(max_depth_options) > 0
    assert np.all(np.array(max_depth_options) > 0)
    if sample_weight_train is not None:
        assert len(sample_weight_train) == len(y_train)
    if sample_weight_valid is not None:
        assert len(sample_weight_valid) == len(y_valid)
    
    this_start_time = time.time()
    hyperparam_dicts = [{'n_estimators': n_estimators,
                         'max_depth'   : max_depth}
                        for n_estimators, max_depth 
                        in product(n_estimators_options, max_depth_options)]
    with mp.get_context('spawn').Pool(processes=min(len(hyperparam_dicts), mp.cpu_count())) as pool:
        models, times = zip(*pool.map(partial(train_single_xgboost, 
                                              X_train = X_train,
                                              y_train = y_train),
                                      hyperparam_dicts))
    for hyperparam_dict_idx in range(len(hyperparam_dicts)):
        log_or_print('Time to train XGBoost classifier n_estimators=' + str(hyperparam_dicts[hyperparam_dict_idx]['n_estimators']) 
                     + ', max_depth=' + str(hyperparam_dicts[hyperparam_dict_idx]['max_depth']) + ': ' 
                     + str(times[hyperparam_dict_idx]) + ' seconds',
                     logger)
    log_or_print('Time to train all XGBoost classifiers: ' + str(time.time() - this_start_time) + ' seconds',
                 logger)
    
    best_valid_auc       = -1
    for hyperparam_dict_idx in range(len(hyperparam_dicts)):
        this_start_time  = time.time()
        n_estimators     = hyperparam_dicts[hyperparam_dict_idx]['n_estimators']
        max_depth        = hyperparam_dicts[hyperparam_dict_idx]['max_depth']
        model            = models[hyperparam_dict_idx]
        valid_pred       = model.predict_proba(X_valid)[:, 1]
        valid_auc        = roc_auc_score(y_valid, 
                                         valid_pred,
                                         sample_weight = sample_weight_valid)
        log_or_print('Valid AUC for n_estimators=' + str(n_estimators) + ', max_depth=' + str(max_depth) 
                     + ': {0:.4f}'.format(valid_auc),
                     logger)
        if valid_auc > best_valid_auc:
            best_valid_auc    = valid_auc
            best_model        = model
            best_n_estimators = n_estimators
            best_max_depth    = max_depth
        log_or_print('Time to evaluate XGBoost classifier with n_estimators=' + str(n_estimators) 
                     + ', max_depth=' + str(max_depth) + ': ' + str(time.time() - this_start_time) + ' seconds',
                     logger)

    log_or_print('Best n_estimators: ' + str(best_n_estimators),
                 logger)
    log_or_print('Best max_depth: ' + str(best_max_depth),
                 logger)
    log_or_print('Best valid AUC: {0:.4f}'.format(best_valid_auc),
                 logger)
    return best_model
    
def save_xgboost(model,
                 file_header,
                 logger = None):
    '''
    Save XGBoost classifier with n_estimators and max_depth in a separate file
    @param model: xgboost XGBClassifier
    @param file_header: str, path to save file, .pkl will be appended for model, _params.json for hyper-parameters
    @param logger: logger, for INFO messages, if None will print to stdout
    @return: None
    '''
    this_start_time  = time.time()
    hyperparams_dict = {'n_estimators': model.n_estimators,
                        'max_depth'   : model.max_depth}
    with open(file_header + '_params.json', 'w') as f:
        json.dump(hyperparams_dict, f)
    model.save_model(file_header + '.pkl')
    log_or_print('Time to save XGBoost classifier: ' + str(time.time() - this_start_time) + ' seconds',
                 logger)
    
def load_xgboost(file_header,
                 logger = None):
    '''
    Load XGBoost classifier
    @param file_header: str, path to file with .pkl appended for model and _params.json appended for hyper-parameters
    @param logger: logger, for INFO messages, if None will print to stdout
    @return: xgboost XGBClassifier
    '''
    this_start_time = time.time()
    with open(file_header + '_params.json', 'r') as f:
        hyperparams_dict = json.load(f)
    model = XGBClassifier(n_estimators = hyperparams_dict['n_estimators'],
                          max_depth    = hyperparams_dict['max_depth'])
    model.load_model(file_header + '.pkl')
    log_or_print('Time to load XGBoost classifier: ' + str(time.time() - this_start_time) + ' seconds',
                 logger)
    return model