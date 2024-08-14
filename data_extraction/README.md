# Data extraction for large-scale study

We build a pipeline to extract data for a large-scale study following the guidelines in Appendix B.

Pre-requisites:
- postgres database containing `cdm` schema in OMOP CDM v6 format
- pip installed version of `omop-learn` package in the same parent directory as this repo

## Preparing for large-scale data extraction

To clean the lab measurements:
- To identify measurements with value 0 that should be null and replace them with null, first get a preliminary version of the reference ranges by running `python3 replace_zeros_with_null_in_measurement.py --create_table=range_{direction} --version=[int]` to produce `cdm_measurement_aux.measurement_{direction}_references`, a table with the most likely reference range for each measurement concept. Set `{direction}` to `low` and `high`.
- Run `python3 replace_zeros_with_null_in_measurement.py --create_table=nulls_replacing_zero --version=[int]` to produce `cdm_measurement_aux.measurement_with_nulls_replacing_zero`, a replicate of the measurement table with nulls replacing zeros that likely should be null.
- Index the newly created tables by running `\i measurement_with_nulls_replacing_zero_indexes.sql` in postgres.
- To load references from clinical sources, run `python3 load_lab_references.py`
- To create cohort and gender-specific reference ranges and standardize references among the top 100 most frequent lab measurements, run `python3 create_lab_reference_tables.py --direction=low` or `high`. If the cohort and gender-specific reference ranges tables have already been created, add `--standardize_only` to only standardize the measurements.
- To drop non-standard measurements, run `python3 create_standardized_measurement_table.py`

To create some preliminary tables in the database:
- To create the prediction date tables, run `python3 create_prediction_date_tables.py`
- To create the race and ethnicity concept tables, run `python3 create_race_and_ethnicity_tables.py`
- To create the general cohort tables, run `python3 extract_data.py --produce_cohort_only=1_year` and `3_years`

To define the most frequent initial diagnoses, most frequent abnormal lab measurements, and groups of procedure codes:
- To identify the 100 most frequent outcomes in the cohort for the 3 categories, run `python3 extract_top_condition_outcomes.py`, `python3 extract_top_procedure_outcomes.py`, and `python3 extract_top_lab_outcomes.py`. Copy the condition and lab output files to new files ending in `_abbreviated.txt` instead of `.txt`. In these new files, abbreviate the concept names for plot titles in the condition and lab output files.
- Condition and lab outcomes can be defined by individual concept IDs. To automate condition or lab outcomes, run `python3 write_automated_data_extraction_scripts.py --outcome_name=` with `condition` or `lab`
- To extract procedure groups, run `python3 create_procedure_cohort_count_table.py` and then `python3 extract_procedure_groupings.py` with the following arguments:
    - `--include=`: a comma-separated list of phrases where a concept with any of these phrases is included
    - `--exclude=`: a comma-separated list of phrases where a concept with any of these phrases is excluded
    - `--group_name=`: name of group for files
    - `--group_name_readable=`: name of group for plot titles
    - To replicate our procedure outcomes, run `./run_procedure_group_extraction.sh` and `python3 write_automated_procedure_data_extraction_scripts.py`

## Running automated data extraction

To run automated data extraction:
- To extract data for a single outcome, run `extract_data.py` with the following arguments:
    - `--outcome=`: `eol`, `condition`, `procedure`, `lab`, `lab_group`
    - `--outcome_id=`: concept ID for condition, procedure, or measurement, comma-separated list allowed for procedures or lab group
    - `--direction=`: `low` or `high` for labs or lab groups, whether we are predicting outcomes above or below range
    - `--omit_features`: if only extracting outcomes
    - `--finalize`: if creating final dataset when features have already been extracted (use in conjunction with omit_features)
    - `--outcome_name=`: name for procedure group, no spaces allowed since using for file name
    - `--outcome_name_readable=`: name of condition, procedure, or lab outcome for plot title
    - `--debug_size`: use to specify limited cohort size for debugging
    - `--cohort_plot_only`: use flag to plot the cohort size and outcome frequency after data has been extracted
    - `--feature_windows`: specify comma-separated list of window lengths in days for features, default: 30
    - `--fold`: specify which data fold to extract, default: 0
    - Note: `extract_omop_data.py` contains the methods called in `extract_data.py`
- After a single condition outcome and a single procedure/lab outcome has been run to generate the shared files, run each `./data_extraction_scripts/run_nonstationarity_check_for_{outcome_name}_outcomes_batch{idx}.sh` to extract data for each outcome.

## Miscellaneous

To compute statistics referenced in the paper:
- To get some cohort statistics, run `python3 compute_cohort_statistics.py`
- To plot frequencies of different eGFR labs over time (reproduce Figure 6 in our paper), run `python3 plot_egfr_label_shift.py`
- To examine some non-stationary lab outcomes, run `python3 examine_nonstationary_lab_outcomes.py` with the optional flag `--order_rates` to examine lab order rates instead of outcome frequencies.
    
For reference, the following tables will be created in the database:
- `prediction_dates_1_year` and `prediction_dates_3_years`: prediction dates from 2014 to 2020 for the 1 year table and 2016 to 2020 for the 3 year table
- `race_concepts` and `ethnicity_concepts`: concept IDs and names for race and ethnicity
- `omop_cohort_1_year` and `omop_cohort_3_years`: Patients in each cohort, where 1 and 3 years specify number of years or prior observation required
- `monthly_eligibility_1_year` and `monthly_eligibility_3_years`: Each month a patient is eligible to be included
- `cohort_procedure_outcome_counts`: counts of procedures in the cohort
- `measurement_references_from_sources`, `measurement_unit_specific_references_from_sources`, `measurement_units_to_drop`, and `measurements_out_of_range`: information for standardizing measurements from clinical sources
- `measurement_age_gender_specific_high_references` and `measurement_age_gender_specific_low_references`: references derived from measurements table
 - `measurement_age_gender_specific_standardized_high_references` and `measurement_age_gender_specific_standardized_low_references`: references derived and standardized
- outcome tables: outcomes for each patient and month