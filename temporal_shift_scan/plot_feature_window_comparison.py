import sys
import json
import matplotlib.pyplot as plt
from os.path import dirname, abspath, join

sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from logging_utils import set_up_logger
from nonstationarity_scan_metric_dict_utils import convert_str_to_num_in_metrics_dict

def compute_auc_differences(outcome_dir):
    '''
    Compute differences between AUCs of current and previous model
    @param outcome_dir: str, name of outcome directory
    @return: list of floats, AUC differences
    '''
    metrics_filename = config.experiment_dir + outcome_dir + '/' + outcome_dir + '_test_metrics.json'
    with open(metrics_filename, 'r') as f:
        metrics      = convert_str_to_num_in_metrics_dict(json.load(f)['metrics_dict'])
    auc_diffs        = []
    for year_idx in range(1, 4):
        curr_auc = metrics[year_idx][year_idx]['auc']
        prev_auc = metrics[year_idx - 1][year_idx]['auc']
        auc_diffs.append(curr_auc - prev_auc)
    return auc_diffs

def plot_feature_window_comparison():
    '''
    Plot the AUC gap from using 30-day windowed features vs the AUC gap from using 365-day windowed features
    Color points by outcome
    @return: None
    '''
    condition_outcome_ids    = [254761, 4223659, 77670, 4144111, 312437]
    condition_outcome_names  = ['Cough', 'Fatigue', 'Chest pain', 'GERD', 'Dyspnea']
    condition_outcome_colors = ['red', 'orange', 'green', 'blue', 'purple'] 
    
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
                           figsize = (4.8, 4.8),
                           sharex  = True)
    
    for outcome_idx in range(len(condition_outcome_ids)):
        # compute AUC differences for each window
        outcome_dir_30days  = 'condition_' + str(condition_outcome_ids[outcome_idx]) + '_outcomes_from_all_freq100_logreg'
        auc_diffs_30days    = compute_auc_differences(outcome_dir_30days)
        outcome_dir_365days = 'condition_' + str(condition_outcome_ids[outcome_idx]) \
                            + '_outcomes_from_all_freq100_logreg_365day_features'
        auc_diffs_365days   = compute_auc_differences(outcome_dir_365days)
        
        # plot AUC differences
        ax.scatter(auc_diffs_365days,
                   auc_diffs_30days,
                   label = condition_outcome_names[outcome_idx],
                   c     = condition_outcome_colors[outcome_idx])
        
    # add y = x reference, legend, format plot, save
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax_min = min([xmin, ymin])
    ax_max = max([xmax, ymax])
    ax.plot([ax_min, ax_max],
            [ax_min, ax_max],
            c = 'black')
    ax.set_xlim([ax_min, ax_max])
    ax.set_ylim([ax_min, ax_max])
    ax.legend()
    ax.set_xlabel('365-day AUC difference')
    ax.set_ylabel('30-day AUC difference')
    ax.set_title('Feature window comparison')
    plt.tight_layout()
    plt.savefig(config.experiment_dir + 'feature_window_comparison.pdf')
    
if __name__ == '__main__':
    plot_feature_window_comparison()