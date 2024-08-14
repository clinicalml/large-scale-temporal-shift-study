# Large-Scale Study of Temporal Shift in Health Insurance Claims

Most machine learning models for predicting clinical outcomes are developed using historical data. Yet, even if these models are deployed in the near future, dataset shift over time may result in less than ideal performance. To capture this phenomenon, we consider a task--that is, an outcome to be predicted at a particular time point--to be non-stationary if a historical model is no longer optimal for predicting that outcome. We build an algorithm to test for temporal shift either at the population level or within a discovered sub-population. Then, we construct a meta-algorithm to perform a retrospective scan for temporal shift on a large collection of tasks. Our algorithms enable us to perform the first comprehensive evaluation of temporal shift in healthcare to our knowledge. We create 1,010 tasks by evaluating 242 healthcare outcomes for temporal shift from 2015 to 2020 on a health insurance claims dataset. 9.7% of the tasks show temporal shifts at the population level, and 93.0% have some sub-population affected by shifts. We dive into case studies to understand the clinical implications. Our analysis highlights the widespread prevalence of temporal shifts in healthcare.

## Definition of prediction task and temporal shift

A prediction task is defined by the following properties:
- Binary outcome: Does a patient have a condition, procedure, abnormal lab measurement, or some other outcome in a fixed window starting on the prediction date, e.g. in the next 90 days?
- Features: Does a patient have a condition, procedure, abnormal lab measurement, or some other feature in a window before the prediction date? These are binary features. Continuous features, such as age, can be included as well.
- Model family: Logistic regression, decision tree, random forest, or other type of model
- Prediction dates in 2 time periods: The test will assess shift between the 2 time periods. For example, to assess for shift between 2019 and 2020, we can define the first of each month in 2019 to be the prediction dates in the first time period and the first of each month in 2020 to be the prediction dates in the second time period.
- Sub-population: Assess temporal shift only for patients in the sub-population. This could be the entire population, a pre-defined sub-population, or a sub-population that is discovered by the method in our paper.

A task is affected by temporal shift if the performance of the model trained on data from the second time period is statistically significantly better than the performance of the model trained on data from the first time period when both models are evaluated on held-out data from the second time period.

## Using the code in this repository

The code in this repository can be used to test for temporal shift in a single task and to scan for temporal shift across a large number of tasks. It can be used with insurance claims or EHR datasets that are in the OMOP CDM format described at https://www.ohdsi.org/data-standardization/

To run the code:
1. Modify `config.py` to set the database name, schema, and output directories.
2. Run `conda create --prefix NEWENV --file conda_env_pkgs.txt` to create the conda environment.
3. Follow the steps in the README in the `data_extraction` directory to prepare the dataset.
4. Follow the steps in the README in the `temporal_shift_scan` directory to run our algorithms to test and scan for temporal shift.

The `utils` directory contains supporting functions.