WITH lab_measurements AS ( 
    SELECT person_id, 
           measurement_date 
    FROM {measurement_aux_schema}.measurement_with_nulls_replacing_zero_drop_nonstandard 
    WHERE measurement_concept_id IN ({concept_id}) 
    AND value_as_number IS NOT NULL 
), 
cohort_lab_measurements AS ( 
    SELECT DISTINCT me.person_id, 
           me.end_date 
    FROM {schema_name}.omop_cohort_{window_name}{debug_suffix} c
    JOIN {schema_name}.monthly_eligibility_{window_name} me 
    ON c.person_id = me.person_id
    JOIN lab_measurements m 
    ON me.person_id = m.person_id 
    AND me.end_date <= m.measurement_date 
    AND me.end_date + 3 * INTERVAL '1 month' >= m.measurement_date 
), 
cohort_lab_counts AS ( 
    SELECT end_date, 
           COUNT(*) AS count 
    FROM cohort_lab_measurements 
    GROUP BY end_date 
) 
SELECT p.prediction_date, 
       COALESCE(c.count, 0) AS count 
FROM {schema_name}.prediction_dates_1_year p 
LEFT JOIN cohort_lab_counts c 
ON DATE(p.prediction_date) = c.end_date 
ORDER BY p.prediction_date;