CREATE TABLE {schema_name}.ethnicity_concepts AS
WITH ethnicity_concept_ids AS (
    SELECT DISTINCT p.ethnicity_concept_id
    FROM cdm.person p
)
SELECT e.ethnicity_concept_id,
       c.concept_name
FROM ethnicity_concept_ids e
JOIN cdm.concept c
ON e.ethnicity_concept_id = c.concept_id;