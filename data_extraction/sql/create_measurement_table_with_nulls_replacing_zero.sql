/*********************************************************************************
Creates measurement table with nulls in place of zeros that are unlikely to be valid results
Zero is considered null if no negative values
AND (reference range low is at least threshold
     OR (reference range low is undefined 
         AND average non-zero value is at least threshold))
********************************************************************************/

CREATE TABLE {measurement_aux_schema}.measurement_with_nulls_replacing_zero AS
WITH measurement_concepts AS ( 
    SELECT DISTINCT measurement_concept_id AS concept_id
    FROM cdm.measurement
), 
measurements_with_negatives AS ( -- zero is null if not exists in here
    SELECT DISTINCT measurement_concept_id AS concept_id
    FROM cdm.measurement
    WHERE value_as_number IS NOT NULL
    AND value_as_number < 0
),
measurements_with_high_references AS ( -- and (exists in here
    SELECT concept_id
    FROM {measurement_aux_schema}.measurement_low_references
    WHERE range_low >= {threshold}
),
measurements_with_no_references AS (
    SELECT concept_id
    FROM measurement_concepts c
    WHERE NOT EXISTS (
        SELECT 
        FROM {measurement_aux_schema}.measurement_low_references r
        WHERE c.concept_id = r.concept_id
    )
),
measurements_with_no_references_avg AS (
    SELECT c.concept_id,
           AVG(m.value_as_number) AS avg_value
    FROM measurements_with_no_references c
    JOIN cdm.measurement m
    ON c.concept_id = m.measurement_concept_id
    AND m.value_as_number IS NOT NULL
    AND m.value_as_number > 0
    GROUP BY c.concept_id
),
measurements_with_high_avg AS ( -- or exists in here)
    SELECT concept_id
    FROM measurements_with_no_references_avg
    WHERE avg_value >= {threshold}
),
measurements_with_high_val AS (
    SELECT concept_id
    FROM measurements_with_high_references 
    UNION
    SELECT concept_id
    FROM measurements_with_high_avg
),
measurements_with_zero_as_null AS (
    SELECT DISTINCT h.concept_id
    FROM measurements_with_high_val h
    WHERE NOT EXISTS (
        SELECT 
        FROM measurements_with_negatives n
        WHERE h.concept_id = n.concept_id
    )
)
SELECT *
FROM cdm.measurement m
WHERE NOT EXISTS (
    SELECT 
    FROM measurements_with_zero_as_null z
    WHERE m.measurement_concept_id = z.concept_id
)
UNION
SELECT *
FROM cdm.measurement m
WHERE EXISTS (
    SELECT 
    FROM measurements_with_zero_as_null z
    WHERE m.measurement_concept_id = z.concept_id
)
AND (m.value_as_number > 0
     OR m.value_as_number IS NULL)
UNION
SELECT m.measurement_id,
       m.person_id,
       m.measurement_concept_id,
       m.measurement_date,
       m.measurement_datetime,
       m.measurement_time,
       m.measurement_type_concept_id,
       m.operator_concept_id,
       NULL as value_as_number,
       m.value_as_concept_id,
       m.unit_concept_id,
       m.range_low,
       m.range_high,
       m.provider_id,
       m.visit_occurrence_id,
       m.visit_detail_id,
       m.measurement_source_value,
       m.measurement_source_concept_id,
       m.unit_source_value,
       m.value_source_value
FROM cdm.measurement m
WHERE EXISTS (
    SELECT 
    FROM measurements_with_zero_as_null z
    WHERE m.measurement_concept_id = z.concept_id
)
AND m.value_as_number = 0;