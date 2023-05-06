/*********************************************************************************
Identifies most frequently occurring reference range for each concept for each age range, gender, and unit:
- Reference must occur at least min_ref_count times for that category
- If two references have the same frequency, use lower reference for range low by passing in reference_order=ASC
and higher reference for range high by passing in reference_order=DESC.

If category does not have a reference but is measured, look among similar concepts:
- Similar concept definition:
  - Shares same first num_characters characters 
  - Has average non-zero values within avg_diff_threshold is used.
- Same tie-breaker as above for same frequency references.
- If two similar concepts have the same most frequent reference and frequency, smaller concept ID is listed.

If category still does not have a reference after looking among similar concepts, order for getting references:
- Reference from lower age ranges descending
- Reference from higher age ranges ascending
- Reference from other gender
- Reference from general population and unspecified unit in cdm_measurement_aux
********************************************************************************/

CREATE TABLE {schema_name}.measurement_age_gender_specific_{direction}_references AS
-- age ranges to try: < 30, 30 - 50, 50 - 70, > 70
WITH measurements_with_age_gender AS ( 
    SELECT m.measurement_concept_id AS concept_id, 
           m.unit_source_value,
           p.person_id,
           EXTRACT(YEAR FROM m.measurement_date) - p.year_of_birth AS age,
           p.gender_source_value,
           m.value_as_number,
           m.range_{direction}
    FROM cdm.measurement m 
    JOIN cdm.person p
    ON m.person_id = p.person_id
    WHERE p.year_of_birth IS NOT NULL
    AND m.measurement_date IS NOT NULL
),
measurements_with_age_categories_gender AS (
    SELECT concept_id,
           unit_source_value,
           person_id,
           CASE WHEN age <= 30
                THEN 'range 1: <= 30'
                WHEN age <= 50
                THEN 'range 2: 31 - 50'
                WHEN age <= 70
                THEN 'range 3: 51 - 70'
                ELSE 'range 4: > 70'
           END AS age_range,
           gender_source_value,
           value_as_number,
           range_{direction}
    FROM measurements_with_age_gender
),
measurements_age_categories_gender_need_reference AS (
    SELECT DISTINCT concept_id,
           unit_source_value,
           age_range,
           gender_source_value
    FROM measurements_with_age_categories_gender
),
measurement_reference_counts AS (
    SELECT concept_id,
           unit_source_value,
           age_range,
           gender_source_value,
           range_{direction},
           COUNT(*) AS reference_count
    FROM measurements_with_age_categories_gender
    WHERE range_{direction} IS NOT NULL
    GROUP BY concept_id,
             unit_source_value,
             age_range,
             gender_source_value,
             range_{direction}
),
most_frequent_references AS (
    SELECT concept_id,
           unit_source_value,
           age_range,
           gender_source_value,
           FIRST_VALUE(range_{direction})
           OVER (
               PARTITION BY concept_id,
                            unit_source_value,
                            age_range,
                            gender_source_value
               ORDER BY reference_count DESC,
                        range_{direction} {reference_order}
           ) range_{direction},
           FIRST_VALUE(reference_count)
           OVER (
               PARTITION BY concept_id,
                            unit_source_value,
                            age_range,
                            gender_source_value
               ORDER BY reference_count DESC,
                        range_{direction} {reference_order}
           ) reference_count
    FROM measurement_reference_counts
    WHERE reference_count >= {min_ref_count}
),
measurement_concepts AS ( 
    SELECT DISTINCT m.concept_id, 
           c.concept_name 
    FROM measurements_age_categories_gender_need_reference m
    JOIN cdm.concept c
    ON m.concept_id = c.concept_id
), 
most_frequent_references_with_names AS (
    SELECT r.concept_id,
           r.unit_source_value,
           c.concept_name,
           r.age_range,
           r.gender_source_value,
           r.range_{direction},
           r.reference_count
    FROM most_frequent_references r
    JOIN measurement_concepts c
    ON r.concept_id = c.concept_id
),
measurements_age_categories_gender_need_reference_with_name AS (
    SELECT m.concept_id,
           m.unit_source_value,
           c.concept_name,
           m.age_range,
           m.gender_source_value
    FROM measurements_age_categories_gender_need_reference m
    JOIN measurement_concepts c
    ON m.concept_id = c.concept_id
),
measurements_without_references AS (
    SELECT m.concept_id,
           m.unit_source_value,
           m.concept_name,
           m.age_range,
           m.gender_source_value
    FROM measurements_age_categories_gender_need_reference_with_name m
    WHERE NOT EXISTS (
        SELECT
        FROM most_frequent_references_with_names r
        WHERE m.concept_id = r.concept_id
        AND m.unit_source_value = r.unit_source_value
        AND m.age_range = r.age_range
        AND m.gender_source_value = r.gender_source_value
    )
),
similar_measurements AS (
    SELECT c.concept_id,
           c.unit_source_value,
           c.concept_name,
           c.age_range,
           c.gender_source_value,
           r.concept_id AS similar_concept_id,
           r.concept_name AS similar_concept_name
    FROM measurements_without_references c
    JOIN most_frequent_references_with_names r
    ON LOWER(LEFT(c.concept_name, {num_characters})) = LOWER(LEFT(r.concept_name, {num_characters}))
    AND c.unit_source_value = r.unit_source_value
    AND c.gender_source_value = r.gender_source_value
    AND c.age_range = r.age_range
),
similar_measurement_concepts AS (
    SELECT DISTINCT concept_id,
           unit_source_value,
           age_range,
           gender_source_value
    FROM similar_measurements
    UNION
    SELECT DISTINCT similar_concept_id AS concept_id,
           unit_source_value,
           age_range,
           gender_source_value
    FROM similar_measurements
),
similar_measurement_averages AS (
    SELECT c.concept_id,
           c.unit_source_value,
           c.age_range,
           c.gender_source_value,
           AVG(m.value_as_number) AS avg_value
    FROM similar_measurement_concepts c
    JOIN measurements_with_age_categories_gender m
    ON c.concept_id = m.concept_id
    AND c.unit_source_value = m.unit_source_value
    AND c.age_range = m.age_range
    AND c.gender_source_value = m.gender_source_value
    WHERE m.value_as_number IS NOT NULL
    GROUP BY c.concept_id,
             c.unit_source_value,
             c.age_range,
             c.gender_source_value
),
similar_measurements_with_avg_diff AS (
    SELECT m.concept_id,
           m.unit_source_value,
           m.concept_name,
           m.age_range,
           m.gender_source_value,
           m.similar_concept_id,
           m.similar_concept_name,
           ABS(c.avg_value - s.avg_value) AS avg_diff
    FROM similar_measurements m
    JOIN similar_measurement_averages c
    ON m.concept_id = c.concept_id
    AND m.unit_source_value = c.unit_source_value
    AND m.age_range = c.age_range
    AND m.gender_source_value = c.gender_source_value
    JOIN similar_measurement_averages s
    ON m.similar_concept_id = s.concept_id
    AND m.unit_source_value = s.unit_source_value
    AND m.age_range = s.age_range
    AND m.gender_source_value = s.gender_source_value
),
similar_measurements_with_avg_diff_filter AS (
    SELECT concept_id,
           unit_source_value,
           concept_name,
           age_range,
           gender_source_value,
           similar_concept_id,
           similar_concept_name
    FROM similar_measurements_with_avg_diff
    WHERE avg_diff <= {avg_diff_threshold}
),
similar_measurement_frequent_references AS (
    SELECT c.concept_id,
           c.unit_source_value,
           c.concept_name,
           c.age_range,
           c.gender_source_value,
           c.similar_concept_id,
           c.similar_concept_name,
           s.range_{direction},
           s.reference_count
    FROM similar_measurements_with_avg_diff_filter c
    JOIN most_frequent_references_with_names s
    ON c.similar_concept_id = s.concept_id
    AND c.unit_source_value = s.unit_source_value
    AND c.age_range = s.age_range
    AND c.gender_source_value = s.gender_source_value
),
similar_measurement_most_frequent_references AS (
    SELECT concept_id,
           unit_source_value,
           concept_name,
           age_range,
           gender_source_value,
           FIRST_VALUE(similar_concept_id)
           OVER (
               PARTITION BY concept_id,
                            unit_source_value,
                            age_range,
                            gender_source_value
               ORDER BY reference_count DESC,
                        range_{direction} {reference_order},
                        similar_concept_id ASC
           ) similar_concept_id,
           FIRST_VALUE(similar_concept_name)
           OVER (
               PARTITION BY concept_id,
                            unit_source_value,
                            age_range,
                            gender_source_value
               ORDER BY reference_count DESC,
                        range_{direction} {reference_order},
                        similar_concept_id ASC
           ) similar_concept_name,
           FIRST_VALUE(range_{direction})
           OVER (
               PARTITION BY concept_id,
                            unit_source_value,
                            age_range,
                            gender_source_value
               ORDER BY reference_count DESC,
                        range_{direction} {reference_order},
                        similar_concept_id ASC
           ) range_{direction}
    FROM similar_measurement_frequent_references
),
measurements_without_similar_references AS (
    SELECT c.concept_id,
           c.unit_source_value,
           c.concept_name,
           c.age_range,
           c.gender_source_value
    FROM measurements_without_references c
    WHERE NOT EXISTS (
        SELECT
        FROM similar_measurement_most_frequent_references r
        WHERE c.concept_id = r.concept_id
        AND c.unit_source_value = r.unit_source_value
        AND c.age_range = r.age_range
        AND c.gender_source_value = r.gender_source_value
    )
),
references_so_far AS (
    SELECT concept_id,
           unit_source_value,
           concept_name,
           age_range,
           gender_source_value,
           -1 AS similar_concept_id,
           '' AS similar_concept_name,
           range_{direction}
    FROM most_frequent_references_with_names
    UNION
    SELECT *
    FROM similar_measurement_most_frequent_references
),
references_so_far_with_opposite_gender AS (
    SELECT c.concept_id,
           c.unit_source_value,
           c.concept_name,
           c.age_range,
           c.gender_source_value,
           r.similar_concept_id,
           r.similar_concept_name,
           r.range_{direction},
           1 AS from_opposite_gender
    FROM measurements_without_similar_references c
    JOIN references_so_far r
    ON c.concept_id = r.concept_id
    AND c.unit_source_value = r.unit_source_value
    AND c.age_range = r.age_range
    UNION
    SELECT *,
           0 AS from_opposite_gender
    FROM references_so_far
),
references_age_lag_lead AS (
    SELECT m.concept_id,
           m.unit_source_value,
           m.concept_name,
           m.age_range,
           m.gender_source_value,
           COALESCE(r.similar_concept_id,
                    LAG(r.similar_concept_id, 1)
                    OVER (
                        PARTITION BY m.concept_id,
                                     m.unit_source_value,
                                     m.concept_name,
                                     m.gender_source_value
                        ORDER BY m.age_range ASC
                    ),
                    LAG(r.similar_concept_id, 2)
                    OVER (
                        PARTITION BY m.concept_id,
                                     m.unit_source_value,
                                     m.concept_name,
                                     m.gender_source_value
                        ORDER BY m.age_range ASC
                    ),
                    LAG(r.similar_concept_id, 3)
                    OVER (
                        PARTITION BY m.concept_id,
                                     m.unit_source_value,
                                     m.concept_name,
                                     m.gender_source_value
                        ORDER BY m.age_range ASC
                    ),
                    LEAD(r.similar_concept_id, 1)
                    OVER (
                        PARTITION BY m.concept_id,
                                     m.unit_source_value,
                                     m.concept_name,
                                     m.gender_source_value
                        ORDER BY m.age_range ASC
                    ),
                    LEAD(r.similar_concept_id, 2)
                    OVER (
                        PARTITION BY m.concept_id,
                                     m.unit_source_value,
                                     m.concept_name,
                                     m.gender_source_value
                        ORDER BY m.age_range ASC
                    ),
                    LEAD(r.similar_concept_id, 3)
                    OVER (
                        PARTITION BY m.concept_id,
                                     m.unit_source_value,
                                     m.concept_name,
                                     m.gender_source_value
                        ORDER BY m.age_range ASC
                    )
           ) AS similar_concept_id,
           COALESCE(r.similar_concept_name,
                    LAG(r.similar_concept_name, 1)
                    OVER (
                        PARTITION BY m.concept_id,
                                     m.unit_source_value,
                                     m.concept_name,
                                     m.gender_source_value
                        ORDER BY m.age_range ASC
                    ),
                    LAG(r.similar_concept_name, 2)
                    OVER (
                        PARTITION BY m.concept_id,
                                     m.unit_source_value,
                                     m.concept_name,
                                     m.gender_source_value
                        ORDER BY m.age_range ASC
                    ),
                    LAG(r.similar_concept_name, 3)
                    OVER (
                        PARTITION BY m.concept_id,
                                     m.unit_source_value,
                                     m.concept_name,
                                     m.gender_source_value
                        ORDER BY m.age_range ASC
                    ),
                    LEAD(r.similar_concept_name, 1)
                    OVER (
                        PARTITION BY m.concept_id,
                                     m.unit_source_value,
                                     m.concept_name,
                                     m.gender_source_value
                        ORDER BY m.age_range ASC
                    ),
                    LEAD(r.similar_concept_name, 2)
                    OVER (
                        PARTITION BY m.concept_id,
                                     m.unit_source_value,
                                     m.concept_name,
                                     m.gender_source_value
                        ORDER BY m.age_range ASC
                    ),
                    LEAD(r.similar_concept_name, 3)
                    OVER (
                        PARTITION BY m.concept_id,
                                     m.unit_source_value,
                                     m.concept_name,
                                     m.gender_source_value
                        ORDER BY m.age_range ASC
                    )
           ) AS similar_concept_name,
           COALESCE(r.range_{direction},
                    LAG(r.range_{direction}, 1)
                    OVER (
                        PARTITION BY m.concept_id,
                                     m.unit_source_value,
                                     m.concept_name,
                                     m.gender_source_value
                        ORDER BY m.age_range ASC
                    ),
                    LAG(r.range_{direction}, 2)
                    OVER (
                        PARTITION BY m.concept_id,
                                     m.unit_source_value,
                                     m.concept_name,
                                     m.gender_source_value
                        ORDER BY m.age_range ASC
                    ),
                    LAG(r.range_{direction}, 3)
                    OVER (
                        PARTITION BY m.concept_id,
                                     m.unit_source_value,
                                     m.concept_name,
                                     m.gender_source_value
                        ORDER BY m.age_range ASC
                    ),
                    LEAD(r.range_{direction}, 1)
                    OVER (
                        PARTITION BY m.concept_id,
                                     m.unit_source_value,
                                     m.concept_name,
                                     m.gender_source_value
                        ORDER BY m.age_range ASC
                    ),
                    LEAD(r.range_{direction}, 2)
                    OVER (
                        PARTITION BY m.concept_id,
                                     m.unit_source_value,
                                     m.concept_name,
                                     m.gender_source_value
                        ORDER BY m.age_range ASC
                    ),
                    LEAD(r.range_{direction}, 3)
                    OVER (
                        PARTITION BY m.concept_id,
                                     m.unit_source_value,
                                     m.concept_name,
                                     m.gender_source_value
                        ORDER BY m.age_range ASC
                    )
           ) AS range_{direction},
           r.from_opposite_gender,
           CASE WHEN r.range_{direction} IS NOT NULL
                THEN 1
                ELSE 0
           END AS from_same_age_range
    FROM measurements_age_categories_gender_need_reference_with_name m
    LEFT JOIN references_so_far_with_opposite_gender r
    ON m.concept_id = r.concept_id
    AND m.unit_source_value = r.unit_source_value
    AND m.age_range = r.age_range
    AND m.gender_source_value = r.gender_source_value
),
references_to_get_from_general AS (
    SELECT concept_id,
           unit_source_value,
           concept_name,
           age_range,
           gender_source_value
    FROM references_age_lag_lead
    WHERE range_{direction} IS NULL
)
SELECT concept_id,
       unit_source_value,
       concept_name,
       age_range,
       gender_source_value,
       CASE WHEN similar_concept_id = -1
            THEN NULL
            ELSE similar_concept_id
       END AS similar_concept_id,
       CASE WHEN similar_concept_id = -1
            THEN NULL
            ELSE similar_concept_name
       END AS similar_concept_name,
       range_{direction},
       from_opposite_gender,
       CASE WHEN from_same_age_range = 1
            THEN 0
            ELSE 1
       END AS from_different_age_range,
       0 AS from_general_reference
FROM references_age_lag_lead
WHERE range_{direction} IS NOT NULL
UNION 
SELECT r.concept_id,
       r.unit_source_value,
       r.concept_name,
       r.age_range,
       r.gender_source_value,
       g.similar_concept_id,
       g.similar_concept_name,
       g.range_{direction},
       0 AS from_opposite_gender,
       0 AS from_different_age_range,
       1 AS from_general_reference
FROM references_to_get_from_general r
JOIN cdm_measurement_aux.measurement_{direction}_references g
ON r.concept_id = g.concept_id;

CREATE INDEX idx_{schema_name}_age_gender_{direction}_references
ON {schema_name}.measurement_age_gender_specific_{direction}_references (
    concept_id ASC,
    unit_source_value ASC,
    age_range ASC,
    gender_source_value ASC
);

CREATE INDEX idx_{schema_name}_age_gender_{direction}_references_concept_unit
ON {schema_name}.measurement_age_gender_specific_{direction}_references (
    concept_id ASC,
    unit_source_value ASC
);