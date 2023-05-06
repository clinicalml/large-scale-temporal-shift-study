SELECT p.example_id,
       p.person_id,
       de.drug_concept_id || ' - drug - ' || c.concept_name AS concept_name,
       DATE(de.drug_exposure_start_datetime) AS feature_start_date,
       p.end_date AS person_end_date
FROM {schema_name}.omop_cohort_{window_name}{eol_suffix}{debug_suffix} p
JOIN cdm.drug_exposure de
ON p.person_id = de.person_id
AND de.drug_exposure_start_datetime <= p.end_date
JOIN cdm.concept c
ON c.concept_id = de.drug_concept_id