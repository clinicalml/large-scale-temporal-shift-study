SELECT p.example_id,
       p.person_id,
       person.year_of_birth AS ntmp_val,
       'Year of birth' AS concept_name
FROM {schema_name}.omop_cohort_{window_name}{eol_suffix}{debug_suffix} p
JOIN cdm.person
ON p.person_id = person.person_id