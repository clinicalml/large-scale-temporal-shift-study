/*********************************************************************************
Identifies most frequently occurring reference range for each concept that occurs at least min_ref_count times:
- If two references have the same frequency, use lower reference for range low by passing in reference_order=ASC
and higher reference for range high by passing in reference_order=DESC.
- If concept does not have any frequent reference ranges provided, then most frequent reference range from another concept 
that shares same first num_characters characters and has average non-zero values within avg_diff_threshold is used.
- Same tie-breaker as above for same frequency references.
- If two similar concepts have the same most frequent reference and frequency, smaller concept ID is listed.
********************************************************************************/

CREATE TABLE cdm_measurement_aux.measurement_{direction}_references AS 
WITH measurement_concepts AS ( 
    SELECT DISTINCT c.concept_id, 
           c.concept_name 
    FROM cdm.concept c
    WHERE EXISTS (
        SELECT
        FROM cdm.measurement m
        WHERE m.measurement_concept_id = c.concept_id
    )
), 
measurement_references AS ( 
    SELECT c.concept_id, 
           c.concept_name, 
           m.range_{direction}, 
           COUNT(*) AS reference_count
    FROM measurement_concepts c 
    JOIN cdm.measurement m 
    ON c.concept_id = m.measurement_concept_id
    WHERE m.range_{direction} IS NOT NULL
    GROUP BY c.concept_id,
             c.concept_name,
             m.range_{direction}
),
measurement_most_frequent_references AS (
    SELECT concept_id,
           concept_name,
           FIRST_VALUE(range_{direction})
           OVER (
               PARTITION BY concept_id
               ORDER BY reference_count DESC,
                        range_{direction} {reference_order}
           ) range_{direction}
    FROM measurement_references
    WHERE reference_count >= {min_ref_count}
),
measurements_without_references AS (
    SELECT c.concept_id,
           c.concept_name
    FROM measurement_concepts c
    WHERE NOT EXISTS (
        SELECT
        FROM measurement_most_frequent_references r
        WHERE c.concept_id = r.concept_id
    )
),
similar_measurements AS (
    SELECT c.concept_id,
           c.concept_name,
           r.concept_id AS similar_concept_id,
           r.concept_name AS similar_concept_name
    FROM measurements_without_references c
    JOIN measurement_most_frequent_references r
    ON LOWER(LEFT(c.concept_name, {num_characters})) = LOWER(LEFT(r.concept_name, {num_characters}))
),
similar_measurement_concepts AS (
    SELECT DISTINCT concept_id
    FROM similar_measurements
    UNION
    SELECT DISTINCT similar_concept_id AS concept_id
    FROM similar_measurements
),
similar_measurement_averages AS (
    SELECT c.concept_id,
           AVG(m.value_as_number) AS avg_value
    FROM similar_measurement_concepts c
    JOIN cdm.measurement m
    ON c.concept_id = m.measurement_concept_id
    AND m.value_as_number IS NOT NULL
    AND m.value_as_number != 0
    GROUP BY c.concept_id
),
similar_measurements_with_avg_diff AS (
    SELECT m.concept_id,
           m.concept_name,
           m.similar_concept_id,
           m.similar_concept_name,
           ABS(c.avg_value - s.avg_value) AS avg_diff
    FROM similar_measurements m
    JOIN similar_measurement_averages c
    ON m.concept_id = c.concept_id
    JOIN similar_measurement_averages s
    ON m.similar_concept_id = s.concept_id
),
similar_measurements_with_avg_diff_filter AS (
    SELECT concept_id,
           concept_name,
           similar_concept_id,
           similar_concept_name
    FROM similar_measurements_with_avg_diff
    WHERE avg_diff <= {avg_diff_threshold}
),
similar_measurement_frequent_references AS (
    SELECT c.concept_id,
           c.concept_name,
           c.similar_concept_id,
           c.similar_concept_name,
           s.range_{direction},
           r.reference_count
    FROM similar_measurements_with_avg_diff_filter c
    JOIN measurement_most_frequent_references s
    ON c.similar_concept_id = s.concept_id
    JOIN measurement_references r
    ON r.concept_id = s.concept_id
    AND r.range_{direction} = s.range_{direction}
),
similar_measurement_most_frequent_references AS (
    SELECT concept_id,
           concept_name,
           FIRST_VALUE(similar_concept_id)
           OVER (
               PARTITION BY concept_id
               ORDER BY reference_count DESC,
                        range_{direction} {reference_order},
                        similar_concept_id ASC
           ) similar_concept_id,
           FIRST_VALUE(similar_concept_name)
           OVER (
               PARTITION BY concept_id
               ORDER BY reference_count DESC,
                        range_{direction} {reference_order},
                        similar_concept_id ASC
           ) similar_concept_name,
           FIRST_VALUE(range_{direction})
           OVER (
               PARTITION BY concept_id
               ORDER BY reference_count DESC,
                        range_{direction} {reference_order},
                        similar_concept_id ASC
           ) range_{direction}
    FROM similar_measurement_frequent_references
) 
SELECT concept_id,
       concept_name,
       NULL AS similar_concept_id,
       NULL AS similar_concept_name,
       range_{direction}
FROM measurement_most_frequent_references
UNION
SELECT concept_id,
       concept_name,
       similar_concept_id,
       similar_concept_name,
       range_{direction}
FROM similar_measurement_most_frequent_references;