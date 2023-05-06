import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from os.path import dirname, abspath, join

sys.path.append(dirname(dirname(abspath(__file__))))
import config

def plot_hypothesis_test_results(summary_dir):
    '''
    Create a plot with 2 subfigures stacked on top of each other:
    1. Number of non-stationary outcomes of each type from our algorithm and baseline
    2. Number of non-stationary sub-populations of each outcome type
    Legend will go underneath both plots
    @param summary_dir: str, path to multiple hypothesis testing results
    @return: None
    '''
    outcome_counts_df          = pd.read_csv(summary_dir 
                                             + 'multiple_hypothesis_testing_clinical_signif_counts.csv')
    baseline_outcome_counts_df = pd.read_csv(summary_dir 
                                             + 'baseline_multiple_hypothesis_testing_clinical_signif_counts.csv')
    subpop_outcome_counts_df   = pd.read_csv(summary_dir 
                                             + 'multiple_hypothesis_testing_subpopulation_clinical_signif_counts.csv')
    
    total_counts   = {'Conditions': 100,
                      'Labs'      : 100,
                      'Procedures': 42}
    
    outcome_counts_plot_df_cols       = ['Algorithm', 'Outcome', 'Year', 'Percentage of tasks']
    outcome_counts_plot_algorithm_col = 13 * ['Our algorithm'] \
                                      + 13 * ['Baseline']
    outcome_counts_plot_outcome_col   = 2 * (  3 * ['Conditions'] \
                                             + 5 * ['Labs'] \
                                             + 5 * ['Procedures'])
    outcome_counts_plot_year_col      = 2 * (  list(range(2018, 2021)) \
                                             + list(range(2016, 2021)) \
                                             + list(range(2016, 2021)))
    year_cols                         = list(map(str, range(2016, 2021)))
    outcome_counts_plot_count_col     = outcome_counts_df[year_cols].values.flatten().tolist()[2:] \
                                      + baseline_outcome_counts_df[year_cols].values.flatten().tolist()[2:]
    outcome_counts_plot_prop_col      \
        = [100 * outcome_counts_plot_count_col[i]/float(total_counts[outcome_counts_plot_outcome_col[i]])
           for i in range(len(outcome_counts_plot_count_col))]
    outcome_counts_plot_df \
        = pd.DataFrame(data    = {'Algorithm'        : outcome_counts_plot_algorithm_col,
                                  'Outcome'          : outcome_counts_plot_outcome_col,
                                  'Year'             : outcome_counts_plot_year_col,
                                  'Percentage of tasks': outcome_counts_plot_prop_col},
                       columns = outcome_counts_plot_df_cols)
    
    subpop_outcome_counts_plot_count_col = subpop_outcome_counts_df[year_cols].values.flatten().tolist()[2:]
    subpop_outcome_counts_plot_prop_col  \
        = [100 * subpop_outcome_counts_plot_count_col[i]/float(total_counts[outcome_counts_plot_outcome_col[i]])
           for i in range(len(subpop_outcome_counts_plot_count_col))]
    subpop_outcome_counts_plot_df_cols   = ['Algorithm', 'Outcome', 'Year', 'Percentage of tasks']
    subpop_outcome_counts_plot_df \
        = pd.DataFrame(data    = {'Algorithm'        : outcome_counts_plot_algorithm_col[:13],
                                  'Outcome'          : outcome_counts_plot_outcome_col[:13],
                                  'Year'             : outcome_counts_plot_year_col[:13],
                                  'Percentage of tasks': subpop_outcome_counts_plot_prop_col},
                       columns = subpop_outcome_counts_plot_df_cols)
    
    plt.clf()
    plt.rc('font', 
           family = 'serif', 
           size   = 14)
    plt.rc('xtick', 
           labelsize = 12)
    plt.rc('ytick', 
           labelsize = 12)
    fig, ax = plt.subplots(nrows   = 2,
                           ncols   = 1,
                           figsize = (6.4, 6.4),
                           sharex  = True)
    sns.lineplot(data        = outcome_counts_plot_df,
                 x           = 'Year',
                 y           = 'Percentage of tasks',
                 hue         = 'Outcome',
                 hue_order   = ['Conditions', 'Labs', 'Procedures'],
                 style       = 'Algorithm',
                 style_order = ['Our algorithm', 'Baseline'],
                 ax          = ax[0])
    sns.lineplot(data        = subpop_outcome_counts_plot_df,
                 x           = 'Year',
                 y           = 'Percentage of tasks',
                 hue         = 'Outcome',
                 hue_order   = ['Conditions', 'Labs', 'Procedures'],
                 style       = 'Algorithm',
                 style_order = ['Our algorithm'],
                 ax          = ax[1])
    
    handles, labels = ax[0].get_legend_handles_labels()
    label_order     = [1,5,2,6,3]
    handles_ordered = [handles[i] for i in label_order]
    labels_ordered  = [labels[i]  for i in label_order]
    ax[0].get_legend().remove()
    ax[1].legend(handles_ordered, 
                 labels_ordered, 
                 loc            = 'upper center', 
                 bbox_to_anchor = (0.435, -0.22),
                 ncol           = 3)
    ax[0].set_title('Non-stationary outcomes')
    ax[1].set_title('Non-stationary sub-populations')
    ax[1].set_xlim([2016, 2020])
    ax[1].set_xticks(ticks = range(2016, 2021))
    ax[0].set_ylim([0,40])
    ax[1].set_ylim([0,100])
    plt.tight_layout()
    plt.subplots_adjust(hspace = 0.2)
    plt.savefig(summary_dir + 'outcome_proportions.pdf')
    
if __name__ == '__main__':
    
    summary_name = 'experiments_selected_with_multiple_hypothesis_testing'
    summary_dir  = config.experiment_dir + summary_name + '/'
    
    plot_hypothesis_test_results(summary_dir)