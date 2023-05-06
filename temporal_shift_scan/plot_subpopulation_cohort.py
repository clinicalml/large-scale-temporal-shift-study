import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from os.path import dirname, abspath, join
sys.path.append(join(dirname(dirname(abspath(__file__))), 'data_extraction'))
from extract_omop_data import plot_cohort_size_and_outcome_freq

def plot_subpopulation_cohort_and_outcome_frequency(config_dict,
                                                    X_csr_all_years,
                                                    Y_all_years,
                                                    region_names_dict,
                                                    get_region_indicators,
                                                    logger,
                                                    overwrite = False):
    '''
    Plot cohort size and outcome frequency summed across all data splits
    for patients in sub-population
    @param config_dict: dict mapping str to int or str, settings such as file paths, plot titles, number of years
    @param X_csr_all_years: dict, map str data split to list of csr matrices, covariates for each year with all features
    @param Y_all_years: dict, map str data split to list of np arrays, outcomes for each year
    @param region_names_dict: dict mapping str to str, names of region and not in region for plot titles and file names
    @param get_region_indicators: function, takes in X and Y, outputs indicators or probabilities 
                                  for whether sample is in region,
                                  if probabilities, samples will be weighted by how likely they are in region
    @param logger: logger, for INFO messages
    @param overwrite: bool, overwrites existing subpopulation cohort plots if True
    @return: None
    '''
    assert X_csr_all_years.keys() == Y_all_years.keys()
    if config_dict['outcome'] in {'lab', 'lab_group'}:
        if config_dict['outcome'] == 'lab':
            lab_outcome_id   = config_dict['outcome_id']
        else:
            lab_outcome_id   = config_dict['outcome_ids']
        eligibility_time = config_dict['eligibility_time'].replace(' ', '_')
        if len(config_dict['debug_suffix']) > 0:
            debug_size   = int(config_dict['debug_suffix'][len('_debug'):])
        else:
            debug_size   = None
    else:
        lab_outcome_id   = None
        eligibility_time = None
        debug_size       = None
    
    region_cohort_plot_fileheader  = config_dict['output_dir'] + config_dict['cohort_plot_header'] + '_' \
                                   + region_names_dict['region_file_name'] + '_cohort_size_outcome_freq'
    region_cohort_plot_filename    = region_cohort_plot_fileheader + '.pdf'
    if (not overwrite) and os.path.exists(region_cohort_plot_filename):
        return
    
    # get region indicators
    start_time                 = time.time()
    interaction_terms_dir      = config_dict['tmp_data_dir'] + config_dict['experiment_name'] + '/'
    if not os.path.exists(interaction_terms_dir):
        os.makedirs(interaction_terms_dir)
    interaction_terms_header   = interaction_terms_dir + 'interactions_'
    data_splits                = list(Y_all_years.keys())
    region_indicators          = {data_split: [get_region_indicators(X_csr_all_years[data_split][idx_year],
                                                                     Y_all_years[data_split][idx_year],
                                                                     interaction_terms_header + data_split + '_year'
                                                                     + str(config_dict['starting_year'] + idx_year) 
                                                                     + '.hf5')
                                               for idx_year in range(config_dict['num_years'])]
                                  for data_split in data_splits}
    if not np.all(np.logical_or(region_indicators[data_splits[0]][0] == 0,
                                region_indicators[data_splits[0]][0] == 1)):
        # plotting method does not handle weighted cohorts
        region_indicators      = {data_split: [np.around(region_indicators[data_split][idx_year])
                                               for idx_year in range(config_dict['num_years'])]
                                  for data_split in data_splits}
    logger.info('Computed region indicators for sub-population cohort plot in ' 
                + str(time.time() - start_time) + ' seconds')
    
    # get cohort plot for comparison
    start_time                    = time.time()
    cohort_plot_header            = config_dict['outcome_specific_data_dir'] + config_dict['cohort_plot_header'] \
                                  + '_cohort_size_outcome_freq'
    cohort_plot_title             = config_dict['outcome_as_feat']
    fig, ax                       = plot_cohort_size_and_outcome_freq(cohort_plot_header,
                                                                      cohort_plot_title,
                                                                      Y_all_years,
                                                                      config_dict['starting_year'],
                                                                      logger,
                                                                      lab_outcome_id   = lab_outcome_id,
                                                                      overwrite        = overwrite,
                                                                      eligibility_time = eligibility_time,
                                                                      debug_size       = debug_size)
    logger.info('Plotted cohort for sub-population cohort plots in ' + str(time.time() - start_time) + ' seconds')
    
    # plot inside sub-population
    start_time                    = time.time()
    region_cohort_plot_title      = config_dict['outcome_as_feat'] + ' in ' + region_names_dict['region_name_short']
    region_Y_all_years            = {data_split: [Y_all_years[data_split][idx_year]
                                                  [np.nonzero(region_indicators[data_split][idx_year])[0]]
                                                  for idx_year in range(config_dict['num_years'])]
                                     for data_split in data_splits}
    fig, ax                       = plot_cohort_size_and_outcome_freq(region_cohort_plot_fileheader,
                                                                      region_cohort_plot_title,
                                                                      region_Y_all_years,
                                                                      config_dict['starting_year'],
                                                                      logger,
                                                                      lab_outcome_id   = lab_outcome_id,
                                                                      overwrite        = overwrite,
                                                                      eligibility_time = eligibility_time,
                                                                      debug_size       = debug_size,
                                                                      fig              = fig,
                                                                      ax               = ax,
                                                                      cohort_name      = 'Sub-population',
                                                                      linestyle        = '--')
    plt.close(fig)
    logger.info('Plotted sub-population ' + region_cohort_plot_title + ' cohort at ' + region_cohort_plot_filename
                + ' in ' + str(time.time() - start_time) + ' seconds')