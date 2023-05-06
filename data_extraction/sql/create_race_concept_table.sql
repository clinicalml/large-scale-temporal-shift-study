CREATE TABLE {schema_name}.race_concepts AS
WITH race_concept_ids AS (
    SELECT DISTINCT p.race_concept_id
    FROM cdm.person p
)
SELECT r.race_concept_id,
       c.concept_name
FROM race_concept_ids r
JOIN cdm.concept c
ON r.race_concept_id = c.concept_id;