SELECT p.example_id,
       p.person_id,
       po.procedure_concept_id || ' - procedure - ' || c.concept_name AS concept_name,
       DATE(po.procedure_datetime) AS feature_start_date,
       p.end_date AS person_end_date
FROM {schema_name}.omop_cohort_{window_name}{eol_suffix}{debug_suffix} p
JOIN cdm.procedure_occurrence po
ON p.person_id = po.person_id
AND po.procedure_datetime <= p.end_date
JOIN cdm.concept c
ON c.concept_id = po.procedure_concept_id