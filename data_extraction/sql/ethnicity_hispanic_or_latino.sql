SELECT p.example_id,
       p.person_id,
       CASE WHEN e.concept_name = 'Hispanic or Latino'
            THEN 1
            ELSE 0
       END AS ntmp_val,
       'Hispanic or Latino' AS concept_name
FROM {schema_name}.omop_cohort_{window_name}{eol_suffix}{debug_suffix} p
JOIN cdm.person
ON p.person_id = person.person_id
JOIN {schema_name}.ethnicity_concepts e
ON person.ethnicity_concept_id = e.ethnicity_concept_id