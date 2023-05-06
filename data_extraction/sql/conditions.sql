SELECT p.example_id,
       p.person_id,
       co.condition_concept_id || ' - condition - ' || c.concept_name AS concept_name,
       DATE(co.condition_start_datetime) AS feature_start_date,
       p.end_date AS person_end_date
FROM {schema_name}.omop_cohort_{window_name}{debug_suffix} p
JOIN cdm.condition_occurrence co
ON p.person_id = co.person_id
AND co.condition_start_datetime <= p.end_date
JOIN cdm.concept c
ON c.concept_id = co.condition_concept_id