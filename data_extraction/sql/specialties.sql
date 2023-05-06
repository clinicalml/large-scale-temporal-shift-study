SELECT p.example_id,
       p.person_id,
       prov.specialty_concept_id || ' - specialty - ' || c.concept_name AS concept_name,
       DATE(vo.visit_start_date) AS feature_start_date,
       p.end_date AS person_end_date
FROM {schema_name}.omop_cohort_{window_name}{eol_suffix}{debug_suffix} p
JOIN cdm.visit_occurrence vo
ON p.person_id = vo.person_id
AND vo.visit_start_date <= p.end_date
JOIN cdm.provider prov
ON prov.provider_id = vo.provider_id
JOIN cdm.concept c
ON c.concept_id = prov.specialty_concept_id