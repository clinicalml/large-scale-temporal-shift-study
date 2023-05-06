SELECT p.example_id,
       p.person_id,
       CASE WHEN r.concept_name = 'Asian'
            THEN 1
            ELSE 0
       END AS ntmp_val,
       'Asian' AS concept_name
FROM {schema_name}.omop_cohort_{window_name}{eol_suffix}{debug_suffix} p
JOIN cdm.person
ON p.person_id = person.person_id
JOIN {schema_name}.race_concepts r
ON person.race_concept_id = r.race_concept_id