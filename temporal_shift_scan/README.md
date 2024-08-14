# Scan for temporal shift

Prior to running these experiments, extract data using `data_extraction`. See its README.

## Testing for non-stationarity in a single task

A single task is a model for predicting an outcome at a particular time point. To test for temporal shift in a single task:
1. Run `python3 run_nonstationarity_check.py` with the following arguments:
    - `--outcome=`: `eol`, `condition`, `procedure`, `lab`, `lab_group`
    - `--outcome_id=`: concept ID for condition, procedure, or measurement, string name for procedures or lab groups
    - `--outcome_ids=`: comma-separated list of concept IDs for lab groups
    - `--direction=`: `low` or `high` for labs, whether we are predicting outcomes above or below range
    - `--features=`: `cond_proc`, `labs`, `drugs`, `all`
    - `--outcome_name=`: name of condition, procedure, or lab outcome for plot title
    - `--model=`: whether to use `logreg`, `dectree`, `forest`, or `xgboost` model for predicting outcome
    - `--debug_size`: specify limited cohort size for debugging
    - `--baseline`: use flag to evaluate gap with baseline instead
    - `--omit_subpopulation`: use flag to omit sub-population analysis
    - `--single_year`: specify year if only checking for non-stationarity in 1 year
    - `--feature_windows`: specify comma-separated list of window lengths in days for features, default: 30
    - `--fold`: specify which data fold to use, default: 0
2. The step above will check for sub-populations that are non-stationary. To check for sub-populations separately, run `python3 run_subpopulation_analysis.py` with the following arguments:
    - `--outcome=`: `eol`, `condition`, `procedure`, `lab`, `lab_group`
    - `--outcome_id=`: concept ID for condition, procedure, or measurement, string name for procedures or lab groups
    - `--outcome_ids=`: comma-separated list of concept IDs for lab groups
    - `--direction=`: `low` or `high` for labs, whether we are predicting outcomes above or below range
    - `--features=`: `cond_proc`, `labs`, `drugs`, `all`
    - `--outcome_name=`: name of condition, procedure, or lab outcome for plot title
    - `--model=`: whether to use `logreg`, `dectree`, `forest`, or `xgboost` model for predicting outcome
    - `--region_model=`: whether to use `logreg`, `dectree`, `forest`, or `xgboost` for region model
    - `--region_identifier=`: whether to use `errors` or `timepoint` as prediction label for region model
    - `--interactions`: use flag if adding (X, Y) interaction terms as covariates in region model
    - `--debug_size`: specify limited cohort size for debugging
    - `--single_feature_regions`: use flag to define regions by features with largest coefficient in logistic regression instead
    of predictions
    - `--baseline`: use flag to evaluate gap with baseline instead
    - `--feature_windows`: specify comma-separated list of window lengths in days for features, default: 30
    - `--fold`: specify which data fold to use, default: 0
    
## Automated large-scale scan

To automate the scan for a large set of outcomes, such as the most frequent conditions and labs as computed in `data_extraction`, and across a large number of years:
1. Run `python3 write_automated_run_experiment_scripts.py` and `python3 write_automated_run_procedure_experiment_scripts.py` with the following arguments:
    - `--outcome=`: `condition`, `lab` (only for the former)
    - `--subsample_sizes=`: comma-separated list of number of samples to use per year
    - `--baseline`: use flag to write scripts for baseline instead
2. The previous script will produce scripts that can be run by `./run_experiment_scripts/run_{baseline_}nonstationarity_check_for_{outcome}_outcomes_batch{num}.sh` This will run step 1 above for multiple tasks and produce csvs listing which years and sub-populations are non-stationary.
3. To create a list of all tasks where temporal shift is detected by Algorithm 2 (with multiple hypothesis testing and the clinical significance criterion), run `python3 summarize_nonstationarity_check_results.py` with the following optional arguments.  This script will also plot Figure 12 showing the distribution of AUC differences among the selected tasks.
    - `--incomplete`: use flag if summarizing before scan is complete. Will skip hypothesis tests that are missing instead of halting.
    - `--exclude=`: comma-separated list of experiment directory names to exclude
4. To create a list of all tasks where the baseline detects temporal shift (also applies multiple hypothesis testing and a clinical significance criterion), run `python3 summarize_baseline_nonstationarity_check_results.py` with the same optional arguments above.
5. To plot the results, run `python3 plot_hypothesis_test_results.py`
6. Results can be found in the experiment directory specified in `config.py`, with summarized results in the sub-directory starting with `experiments_selected_with_multiple_hypothesis_testing`

## Case studies

To examine domain shift in 2020:
1. Run `python3 examine_domain_shift_in_2020.py`
2. Select some features for plotting from `domain_shift_2020_v_2019_plot_features.csv`. Copy the rows to `domain_shift_2020_v_2019_plot_features_cleaned.csv` and modify the `Feature` column to create short readable names for plotting.
3. Run `python3 examine_domain_shift_in_2020.py --plot_features` to reproduce Figure 7 in our paper.
4. To find examples of tasks where conditions with domain shift are important features, run `python3 find_domain_shift_examples.py` with the following arguments:
    - `--num_top_coefs=`: number of top positive coefficients to look for conditions in
5. To examine addressing domain shift by predicting missing features for a specific condition outcome, run `python3 predict_missing_features_for_domain_shift.py`. Add `--oracle` to run the 2020 version
6. To examine addressing domain shift by using 365-day features or only drug features, run `python3 run_nonstationarity_check.py` with `--feature_windows=365` or `--features=drugs`. To check test AUCs from this experiment, run `python3 check_domain_shift_robust_features_test_auc.py` with `--version=365day` or `--version=drugs`
7. To run the importance reweighting baseline, run `python3 run_importance_reweighting.py`

To find examples of conditional shift via coefficient changes or interpretable sub-populations:
1. Run `python3 find_conditional_shift_examples.py` with the following arguments:
    - `--interpretable_subpop=`: specify whether to look for sub-populations at shorter tree branches
    - `--max_depth=`: maximum depth of region leaf in sub-population tree
    - `--coef_change=`: specify whether to look for top coefficients that change signs
    - `--num_top_coefs=`: number of top positive coefficients to look for changes in
    - `--significant_coef_change=`: specify whether to look for coefficients where the confidence 
2. To examine outcome frequencies in feature groups, run `python3 examine_conditional_shifts.py`
3. To address conditional shift by re-calibrating predictions for specific sub-populations, run `python3 recalibrate_subpopulations_for_conditional_shift.py` with the argument `--outcome_name`
4. To address conditional shift by learning models with more robust features, run `python3 select_robust_features_for_conditional_shift.py` with the argument `--outcome_name` and `--statsmodels` if using statsmodels instead of scikit-learn

## Reproducing figures

To reproduce Figure 4 in our paper:
1. Run `python3 plot_nonstationarity_checks_in_paper.py`.

To reproduce Figure 8 in our paper:
1. Run `python3 plot_feature_window_comparison.py`.

To reproduce Figure 10 in our paper:
1. Run `python3 plot_negative_log_likelihood_difference_distribution.py --together`.