import argparse
import sys
import os
from os.path import dirname, abspath, join

sys.path.append(dirname(dirname(abspath(__file__))))
import config

def create_parser():
    '''
    Create an argument parser
    @return: argparse ArgumentParser
    '''
    parser = argparse.ArgumentParser(description='Run an experiment.')
    parser.add_argument('--outcome', 
                        action  = 'store', 
                        type    = str, 
                        help    = 'Specify outcome among eol, condition, procedure, lab, lab_group')
    parser.add_argument('--outcome_id', 
                        action  = 'store', 
                        type    = str, 
                        default = '', 
                        help    = ('Specify ID of condition or lab outcome or string for procedure or lab group. '
                                   'No spaces allowed since used for file names.')
                       )
    parser.add_argument('--outcome_ids',
                        action  = 'store',
                        type    = str,
                        default = '',
                        help    = 'Specify comma-separated list of lab outcome IDs for lab group outcomes.')
    parser.add_argument('--direction',
                        action  = 'store',
                        type    = str,
                        default = '',
                        help    = 'Specify low or high for abnormal lab outcome')
    parser.add_argument('--features', 
                        action  = 'store', 
                        type    = str, 
                        help    = 'Specify features among cond_proc, drugs, labs, all')
    parser.add_argument('--outcome_name',
                        action  = 'store',
                        type    = str,
                        default = '',
                        help    = 'Specify name of condition, procedure, or lab outcome for plot title.')
    parser.add_argument('--model',
                        action  = 'store',
                        type    = str,
                        help    = 'Specify model type for each year: logreg, dectree, forest, or xgboost.')
    parser.add_argument('--debug_size',
                        action  = 'store',
                        type    = int,
                        default = None,
                        help    = 'Specify smaller cohort size for debugging.')
    parser.add_argument('--baseline',
                        action  = 'store_true',
                        default = False,
                        help    = 'Specify whether to perform baseline evaluation instead.')
    parser.add_argument('--omit_subpopulation',
                        action  = 'store_true',
                        default = False,
                        help    = 'Specify whether to omit sub-population analysis.')
    parser.add_argument('--single_year',
                        action  = 'store',
                        type    = int,
                        default = None,
                        help    = 'Specify a single year to run the non-stationarity check for.')
    parser.add_argument('--feature_windows',
                        action  = 'store',
                        type    = str,
                        default = '30',
                        help    = 'Specify comma-separated list of feature window lengths in days.')
    parser.add_argument('--fold',
                        action  = 'store',
                        type    = int,
                        default = 0,
                        help    = ('Specify which fold to use data from: 0, 1, 2, or 3.'))
    return parser

def create_config_dict(args):
    '''
    Create configuration dictionary from arguments
    @param args: arguments from parser
    @return: dict mapping str to ints and strs specifying file paths, plot titles, number of years, and other settings
    '''
    config_dict = dict()
    
    config_dict['data_dir']        = config.outcome_data_dir
    config_dict['outcome']         = args.outcome
    config_dict['outcome_id']      = args.outcome_id
    config_dict['outcome_ids']     = args.outcome_ids
    config_dict['outcome_name']    = args.outcome_name
    config_dict['feature_set']     = args.features
    config_dict['direction']       = args.direction
    config_dict['model']           = args.model
    config_dict['baseline']        = args.baseline
    config_dict['fold']            = args.fold
    config_dict['feature_windows'] = [int(days) for days in args.feature_windows.split(',')]
    if len(config_dict['feature_windows']) == 1 and config_dict['feature_windows'][0] == 30:
        config_dict['feature_window_suffix'] = ''
    else:
        config_dict['feature_window_suffix'] = '_' + '_'.join(map(str, config_dict['feature_windows'])) + 'day_features'
    
    if args.debug_size is not None:
        config_dict['debug_suffix'] = '_debug' + str(args.debug_size)
    else:
        config_dict['debug_suffix'] = ''
    
    feature_plot_titles = {'cond_proc': 'conditions and procedures',
                           'drugs'    : 'drugs',
                           'labs'     : 'labs',
                           'all'      : 'all features'}
    if args.outcome == 'condition':
        # account for smaller cohort size
        config_dict['min_freq']      = 100
        config_dict['num_years']     = 4 # years: 2017 - 2020
        config_dict['starting_year'] = 2017
    else:
        config_dict['min_freq']      = 300
        config_dict['num_years']     = 6 # years: 2015 - 2020
        config_dict['starting_year'] = 2015
    config_dict['all_outputs_dir']   = config.experiment_dir
    if args.single_year is not None:
        # compare that year with the previous year
        assert args.single_year > config_dict['starting_year']
        assert args.single_year < config_dict['starting_year'] + config_dict['num_years']
        config_dict['starting_year_idx'] = args.single_year - config_dict['starting_year'] - 1
        config_dict['num_years']         = 2
    else:
        config_dict['starting_year_idx'] = 0
    if args.outcome == 'eol':
        config_dict['outcome_specific_data_dir'] = config_dict['data_dir'] + 'dataset_eol_outcomes' \
                                                 + config_dict['feature_window_suffix'] + config_dict['debug_suffix'] + '/'
        config_dict['experiment_name']     = 'eol_outcomes_from_' + args.features + '_freq' + str(config_dict['min_freq']) \
                                           + '_' + config_dict['model'] + config_dict['feature_window_suffix'] \
                                           + config_dict['debug_suffix']
        config_dict['plot_title']          = 'Mortality from ' + feature_plot_titles[args.features]
        config_dict['cohort_plot_file']    = 'eol_outcomes' + config_dict['debug_suffix'] + '_cohort_size_outcome_freq.pdf'
    elif args.outcome in {'condition', 'procedure'}:
        config_dict['outcome_specific_data_dir'] = config_dict['data_dir'] + 'dataset_' + args.outcome + '_' + args.outcome_id \
                                                 + '_outcomes' + config_dict['feature_window_suffix'] \
                                                 + config_dict['debug_suffix'] + '/'
        config_dict['experiment_name']     = args.outcome + '_' + args.outcome_id + '_outcomes_from_' + args.features \
                                           + '_freq' + str(config_dict['min_freq']) + '_' + config_dict['model'] \
                                           + config_dict['feature_window_suffix'] + config_dict['debug_suffix']
        config_dict['plot_title']          = args.outcome_name + ' from ' + feature_plot_titles[args.features]
        config_dict['cohort_plot_file']    = args.outcome + '_' + args.outcome_id + '_outcomes' + config_dict['debug_suffix'] \
                                           + '_cohort_size_outcome_freq.pdf'
    else: # args.outcome in {'lab', 'lab_group'}
        config_dict['outcome_specific_data_dir'] = config_dict['data_dir'] + 'dataset_' + args.outcome + '_' + args.outcome_id \
                                                 + '_' + args.direction + '_outcomes' + config_dict['feature_window_suffix'] \
                                                 + config_dict['debug_suffix'] + '/'
        config_dict['experiment_name']     = args.outcome + '_' + args.outcome_id + '_' + args.direction + '_outcomes_from_' \
                                           + args.features + '_freq' + str(config_dict['min_freq']) \
                                           + '_' + config_dict['model'] + config_dict['feature_window_suffix'] \
                                           + config_dict['debug_suffix']
        config_dict['plot_title']          = args.outcome_name + ' ' + args.direction +  ' from ' \
                                           + feature_plot_titles[args.features]
        config_dict['cohort_plot_file']    = args.outcome + '_' + args.outcome_id + '_' + args.direction + '_outcomes' \
                                           + config_dict['debug_suffix'] + '_cohort_size_outcome_freq.pdf'
    
    if args.fold == 0:
        fold_str         = ''
        data_fold_str    = 'fold' + str(args.fold)
        test_fold_str    = 'fold0'
    else:
        fold_str         = '_fold' + str(args.fold) + '_using_fold0_features'
        data_fold_str    = 'fold' + str(args.fold) + '_using_fold0'
        test_fold_str    = 'fold0'
    config_dict['dataset_file_header']   = config_dict['outcome_specific_data_dir'] + data_fold_str + '_freq' \
                                         + str(config_dict['min_freq'])
    config_dict['test_data_file_header'] = config_dict['outcome_specific_data_dir'] + test_fold_str + '_freq' \
                                         + str(config_dict['min_freq'])
    config_dict['output_dir']           = config_dict['all_outputs_dir'] + config_dict['experiment_name'] + fold_str + '/'
    if not os.path.exists(config_dict['output_dir']):
        os.makedirs(config_dict['output_dir'])
    config_dict['output_file_header']   = config_dict['output_dir'] + config_dict['experiment_name'] + '_'
    
    return config_dict