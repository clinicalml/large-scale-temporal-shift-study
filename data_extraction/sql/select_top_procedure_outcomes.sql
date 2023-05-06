SELECT *
FROM {schema_name}.cohort_procedure_outcome_counts
ORDER BY concept_count DESC,
         concept_id ASC
LIMIT {number_outcomes};