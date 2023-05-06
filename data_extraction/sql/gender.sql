SELECT p.example_id,
       p.person_id,
       CASE WHEN c.concept_name =  'MALE'
            THEN 1
            ELSE 0
       END AS ntmp_val,
       'Male' AS concept_name
FROM {schema_name}.omop_cohort_{window_name}{eol_suffix}{debug_suffix} p
JOIN cdm.person
ON p.person_id = person.person_id
JOIN cdm.concept c
ON person.gender_concept_id = c.concept_id