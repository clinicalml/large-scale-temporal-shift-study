WITH cohort_ids AS (
    SELECT DISTINCT person_id
    FROM {schema_name}.monthly_eligibility_1_year
),
cohort_birth_year_genders AS (
    SELECT c.person_id,
           p.year_of_birth,
           p.gender_source_value
    FROM cohort_ids c
    JOIN cdm.person p
    ON c.person_id = p.person_id
),
cohort_outcome_measurements AS (
    SELECT DISTINCT c.person_id, 
           c.end_date, 
           m.measurement_concept_id AS concept_id, 
           m.value_as_number,
           m.unit_source_value,
           EXTRACT(YEAR FROM m.measurement_date) - p.year_of_birth AS age,
           p.gender_source_value
    FROM {schema_name}.monthly_eligibility_1_year c
    JOIN cohort_birth_year_genders p
    ON c.person_id = p.person_id
    JOIN {measurement_aux_schema}.measurement_with_nulls_replacing_zero_drop_nonstandard m 
    ON c.person_id = m.person_id 
    AND c.end_date <= m.measurement_date 
    AND c.end_date + 3 * INTERVAL '1 month' >= m.measurement_date 
    AND m.value_as_number IS NOT NULL
),
cohort_outcome_measurements_age_range AS (
    SELECT person_id,
           end_date,
           concept_id,
           value_as_number,
           unit_source_value,
           CASE WHEN age <= 30
                THEN 'range 1: <= 30'
                WHEN age <= 50
                THEN 'range 2: 31 - 50'
                WHEN age <= 70
                THEN 'range 3: 51 - 70'
                ELSE 'range 4: > 70'
           END AS age_range,
           gender_source_value
    FROM cohort_outcome_measurements
),
cohort_outcome_occurrences AS ( 
    SELECT DISTINCT m.person_id, 
           m.end_date, 
           m.concept_id, 
           'low' AS direction
    FROM cohort_outcome_measurements_age_range m 
    JOIN {measurement_aux_schema}.measurement_age_gender_specific_standardized_low_references r 
    ON m.concept_id = r.concept_id 
    AND m.unit_source_value = r.unit_source_value
    AND m.age_range = r.age_range
    AND m.gender_source_value = r.gender_source_value
    WHERE m.value_as_number < r.range_low 
    UNION 
    SELECT DISTINCT m.person_id, 
           m.end_date, 
           m.concept_id, 
           'high' AS direction 
    FROM cohort_outcome_measurements_age_range m 
    JOIN {measurement_aux_schema}.measurement_age_gender_specific_standardized_high_references r 
    ON m.concept_id = r.concept_id 
    AND m.unit_source_value = r.unit_source_value
    AND m.age_range = r.age_range
    AND m.gender_source_value = r.gender_source_value
    WHERE m.value_as_number > r.range_high 
), 
top_outcome_counts AS ( 
    SELECT concept_id, 
           direction, 
           COUNT(*) AS concept_count 
    FROM cohort_outcome_occurrences 
    GROUP BY concept_id, 
             direction
    ORDER BY concept_count DESC,
             concept_id ASC,
             direction DESC
    LIMIT {number_outcomes}
),
top_outcome_counts_with_units AS (
    SELECT DISTINCT c.*,
           m.unit_source_value
    FROM top_outcome_counts c
    JOIN cohort_outcome_measurements m
    ON c.concept_id = m.concept_id
),
top_outcome_references AS (
    SELECT c.*,
           m1.range_low AS male_reference_age_range_1,
           m2.range_low AS male_reference_age_range_2,
           m3.range_low AS male_reference_age_range_3,
           m4.range_low AS male_reference_age_range_4,
           f1.range_low AS female_reference_age_range_1,
           f2.range_low AS female_reference_age_range_2,
           f3.range_low AS female_reference_age_range_3,
           f4.range_low AS female_reference_age_range_4
    FROM top_outcome_counts_with_units c
    LEFT JOIN {measurement_aux_schema}.measurement_age_gender_specific_standardized_low_references m1 
    ON c.concept_id = m1.concept_id
    AND c.unit_source_value = m1.unit_source_value
    AND m1.gender_source_value = 'M'
    AND m1.age_range = 'range 1: <= 30'
    LEFT JOIN {measurement_aux_schema}.measurement_age_gender_specific_standardized_low_references m2 
    ON c.concept_id = m2.concept_id
    AND c.unit_source_value = m2.unit_source_value
    AND m2.gender_source_value = 'M'
    AND m2.age_range = 'range 2: 31 - 50'
    LEFT JOIN {measurement_aux_schema}.measurement_age_gender_specific_standardized_low_references m3 
    ON c.concept_id = m3.concept_id
    AND c.unit_source_value = m3.unit_source_value
    AND m3.gender_source_value = 'M'
    AND m3.age_range = 'range 3: 51 - 70'
    LEFT JOIN {measurement_aux_schema}.measurement_age_gender_specific_standardized_low_references m4 
    ON c.concept_id = m4.concept_id
    AND c.unit_source_value = m4.unit_source_value
    AND m4.gender_source_value = 'M'
    AND m4.age_range = 'range 4: > 70'
    LEFT JOIN {measurement_aux_schema}.measurement_age_gender_specific_standardized_low_references f1 
    ON c.concept_id = f1.concept_id
    AND c.unit_source_value = f1.unit_source_value
    AND f1.age_range = 'range 1: <= 30'
    AND f1.gender_source_value = 'F'
    LEFT JOIN {measurement_aux_schema}.measurement_age_gender_specific_standardized_low_references f2 
    ON c.concept_id = f2.concept_id
    AND c.unit_source_value = f2.unit_source_value
    AND f2.age_range = 'range 2: 31 - 50'
    AND f2.gender_source_value = 'F'
    LEFT JOIN {measurement_aux_schema}.measurement_age_gender_specific_standardized_low_references f3 
    ON c.concept_id = f3.concept_id
    AND c.unit_source_value = f3.unit_source_value
    AND f3.age_range = 'range 3: 51 - 70'
    AND f3.gender_source_value = 'F'
    LEFT JOIN {measurement_aux_schema}.measurement_age_gender_specific_standardized_low_references f4 
    ON c.concept_id = f4.concept_id
    AND c.unit_source_value = f4.unit_source_value
    AND f4.age_range = 'range 4: > 70'
    AND f4.gender_source_value = 'F'
    WHERE c.direction = 'low'
    UNION
    SELECT c.*,
           m1.range_high AS male_reference_age_range_1,
           m2.range_high AS male_reference_age_range_2,
           m3.range_high AS male_reference_age_range_3,
           m4.range_high AS male_reference_age_range_4,
           f1.range_high AS female_reference_age_range_1,
           f2.range_high AS female_reference_age_range_2,
           f3.range_high AS female_reference_age_range_3,
           f4.range_high AS female_reference_age_range_4
    FROM top_outcome_counts_with_units c
    LEFT JOIN {measurement_aux_schema}.measurement_age_gender_specific_standardized_high_references m1 
    ON c.concept_id = m1.concept_id
    AND c.unit_source_value = m1.unit_source_value
    AND m1.gender_source_value = 'M'
    AND m1.age_range = 'range 1: <= 30'
    LEFT JOIN {measurement_aux_schema}.measurement_age_gender_specific_standardized_high_references m2 
    ON c.concept_id = m2.concept_id
    AND c.unit_source_value = m2.unit_source_value
    AND m2.gender_source_value = 'M'
    AND m2.age_range = 'range 2: 31 - 50'
    LEFT JOIN {measurement_aux_schema}.measurement_age_gender_specific_standardized_high_references m3 
    ON c.concept_id = m3.concept_id
    AND c.unit_source_value = m3.unit_source_value
    AND m3.gender_source_value = 'M'
    AND m3.age_range = 'range 3: 51 - 70'
    LEFT JOIN {measurement_aux_schema}.measurement_age_gender_specific_standardized_high_references m4 
    ON c.concept_id = m4.concept_id
    AND c.unit_source_value = m4.unit_source_value
    AND m4.gender_source_value = 'M'
    AND m4.age_range = 'range 4: > 70'
    LEFT JOIN {measurement_aux_schema}.measurement_age_gender_specific_standardized_high_references f1 
    ON c.concept_id = f1.concept_id
    AND c.unit_source_value = f1.unit_source_value
    AND f1.age_range = 'range 1: <= 30'
    AND f1.gender_source_value = 'F'
    LEFT JOIN {measurement_aux_schema}.measurement_age_gender_specific_standardized_high_references f2 
    ON c.concept_id = f2.concept_id
    AND c.unit_source_value = f2.unit_source_value
    AND f2.age_range = 'range 2: 31 - 50'
    AND f2.gender_source_value = 'F'
    LEFT JOIN {measurement_aux_schema}.measurement_age_gender_specific_standardized_high_references f3 
    ON c.concept_id = f3.concept_id
    AND c.unit_source_value = f3.unit_source_value
    AND f3.age_range = 'range 3: 51 - 70'
    AND f3.gender_source_value = 'F'
    LEFT JOIN {measurement_aux_schema}.measurement_age_gender_specific_standardized_high_references f4 
    ON c.concept_id = f4.concept_id
    AND c.unit_source_value = f4.unit_source_value
    AND f4.age_range = 'range 4: > 70'
    AND f4.gender_source_value = 'F'
    WHERE c.direction = 'high'
)
SELECT r.concept_id,
       c.concept_name,
       r.direction,
       r.concept_count,
       r.unit_source_value,
       r.male_reference_age_range_1,
       r.male_reference_age_range_2,
       r.male_reference_age_range_3,
       r.male_reference_age_range_4,
       r.female_reference_age_range_1,
       r.female_reference_age_range_2,
       r.female_reference_age_range_3,
       r.female_reference_age_range_4
FROM top_outcome_references r
JOIN cdm.concept c 
ON r.concept_id = c.concept_id 
ORDER BY r.concept_count DESC,
         r.concept_id ASC,
         r.direction DESC;