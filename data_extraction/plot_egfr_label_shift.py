import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import sys
from os.path import dirname, abspath, join

sys.path.append(dirname(dirname(abspath(__file__))))
import config
    
def plot_egfr_label_shift():
    '''
    Plot proportion of samples with each eGFR concept taken within 3 months of prediction date
    Place legend underneath
    @return: None
    '''
    outcome_names = ['MDRD among non-blacks',
                     'MDRD among blacks',
                     'MDRD',
                     'CKD-EPI among non-blacks',
                     'CKD-EPI among blacks',
                     'CKD']
    outcome_ids   = [3049187,
                     3053283,
                     3030354,
                     36303797,
                     36306178,
                     40764999]
    
    egfr_df_cols           = ['eGFR formula', 'Adjustment', 'Year', 'Frequency']
    egfr_df_formula_col    = 18 * ['MDRD'] + 18 * ['CKD-EPI']
    egfr_df_adjustment_col = 2 * (  6 * ['For non-African American'] 
                                  + 6 * ['For African American']
                                  + 6 * ['Unspecified'])
    egfr_df_year_col       = 6 * [2015 + i for i in range(6)]
    egfr_df_frequency_col  = []
    
    for outcome_idx in range(len(outcome_ids)):
        outcome_id   = outcome_ids[outcome_idx]
        outcome_name = outcome_names[outcome_idx]
        with open(config.outcome_data_dir + 'dataset_lab_' + str(outcome_id) + '_low_outcomes/lab_' + str(outcome_id) 
                  + '_low_outcomes_cohort_size_outcome_freq.json', 'r') as f:
            outcome_metrics = json.load(f)
        outcome_freqs = np.divide(np.array(outcome_metrics['Cohort with lab']),
                                  np.array(outcome_metrics['Cohort size']))
        egfr_df_frequency_col += outcome_freqs.tolist()
    
    egfr_df = pd.DataFrame(data    = {'eGFR formula': egfr_df_formula_col,
                                      'Adjustment'  : egfr_df_adjustment_col,
                                      'Year'        : egfr_df_year_col,
                                      'Frequency'   : egfr_df_frequency_col},
                           columns = egfr_df_cols)
    
    plt.clf()
    plt.rc('font', 
           family = 'serif', 
           size   = 14)
    plt.rc('xtick', 
           labelsize = 12)
    plt.rc('ytick', 
           labelsize = 12)
    fig, ax = plt.subplots(nrows   = 1,
                           ncols   = 1,
                           figsize = (6.4, 4.8))
    sns.lineplot(data        = egfr_df,
                 x           = 'Year',
                 y           = 'Frequency',
                 hue         = 'Adjustment',
                 hue_order   = ['For non-African American', 'For African American', 'Unspecified'],
                 style       = 'eGFR formula',
                 style_order = ['MDRD', 'CKD-EPI'],
                 ax          = ax)
    ax.set_title('eGFR outcomes')
    ax.set_xlim([2015, 2020])
    ax.set_xticks(range(2015, 2021))
    ax.set_ylim(bottom = 0)
    ax.legend(loc            = 'upper center',
              bbox_to_anchor = (0.435, -0.25),
              ncol           = 2)
    plt.tight_layout()
    plt.savefig(config.outcome_data_dir + 'egfr_label_shift.pdf')
    
if __name__ == '__main__':
    plot_egfr_label_shift()