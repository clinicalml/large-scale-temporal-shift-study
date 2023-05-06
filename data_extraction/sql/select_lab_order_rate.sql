WITH cohort_ids AS (
    SELECT DISTINCT person_id
    FROM {schema_name}.omop_cohort_1_year
),
lab_concept_measurements_available AS (
    SELECT CASE WHEN m.value_as_number IS NOT NULL
                THEN 1
                ELSE 0
           END AS value_available
    FROM cdm.measurement m
    JOIN cohort_ids c
    ON m.person_id = c.person_id
    WHERE m.measurement_concept_id = {concept_id}
    AND m.measurement_date >= DATE('{start_date}')
    AND m.measurement_date <= DATE('{end_date}')
)
SELECT value_available,
       COUNT(*) AS num_measurements
FROM lab_concept_measurements_available
GROUP BY value_available;