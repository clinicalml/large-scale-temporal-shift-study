WITH cohort_ids AS (
    SELECT DISTINCT person_id
    FROM {schema_name}.omop_cohort_{window_name}{eol_suffix}{debug_suffix}
),
cohort_birth_year_genders AS (
    SELECT c.person_id,
           p.year_of_birth,
           p.gender_source_value
    FROM cohort_ids c
    JOIN cdm.person p
    ON c.person_id = p.person_id
),
cohort_measurements AS (
    SELECT p.example_id,
           p.person_id,
           m.measurement_concept_id,
           m.value_as_number,
           m.unit_source_value,
           DATE(m.measurement_datetime) AS feature_start_date,
           p.end_date AS person_end_date
    FROM {schema_name}.omop_cohort_{window_name}{eol_suffix}{debug_suffix} p
    JOIN {measurement_aux_schema}.measurement_with_nulls_replacing_zero_drop_nonstandard m
    ON p.person_id = m.person_id
    AND m.measurement_datetime <= p.end_date
),
measurement_concept_ids AS (
    SELECT DISTINCT measurement_concept_id
    FROM cohort_measurements
),
measurement_concept_names AS (
    SELECT m.measurement_concept_id,
           c.concept_name
    FROM measurement_concept_ids m
    JOIN cdm.concept c
    ON m.measurement_concept_id = c.concept_id
),
cohort_measurements_with_names AS (
    SELECT m.*,
           c.concept_name
    FROM cohort_measurements m
    JOIN measurement_concept_names c
    ON m.measurement_concept_id = c.measurement_concept_id
),
cohort_measurements_with_values AS (
    SELECT *
    FROM cohort_measurements_with_names
    WHERE value_as_number IS NOT NULL
),
cohort_measurements_with_values_age_gender AS (
    SELECT m.*,
           EXTRACT(YEAR FROM feature_start_date) - p.year_of_birth AS age,
           p.gender_source_value
    FROM cohort_measurements_with_values m
    JOIN cohort_birth_year_genders p
    ON m.person_id = p.person_id
),
cohort_measurements_with_values_age_ranges_gender AS (
    SELECT example_id,
           person_id,
           measurement_concept_id,
           concept_name,
           value_as_number,
           unit_source_value,
           feature_start_date,
           person_end_date,
           CASE WHEN age <= 30
                THEN 'range 1: <= 30'
                WHEN age <= 50
                THEN 'range 2: 31 - 50'
                WHEN age <= 70
                THEN 'range 3: 51 - 70'
                ELSE 'range 4: > 70'
           END AS age_range,
           gender_source_value
    FROM cohort_measurements_with_values_age_gender
),
cohort_measurements_with_values_and_references AS (
    SELECT m.example_id,
           m.person_id,
           m.measurement_concept_id,
           m.concept_name,
           m.value_as_number,
           m.unit_source_value,
           m.feature_start_date,
           m.person_end_date,
           lr.range_low,
           hr.range_high
    FROM cohort_measurements_with_values_age_ranges_gender m
    LEFT JOIN {measurement_aux_schema}.measurement_age_gender_specific_standardized_low_references lr
    ON m.measurement_concept_id = lr.concept_id
    AND m.unit_source_value = lr.unit_source_value
    AND m.gender_source_value = lr.gender_source_value
    AND m.age_range = lr.age_range
    LEFT JOIN {measurement_aux_schema}.measurement_age_gender_specific_standardized_high_references hr
    ON m.measurement_concept_id = hr.concept_id
    AND m.unit_source_value = hr.unit_source_value
    AND m.gender_source_value = hr.gender_source_value
    AND m.age_range = hr.age_range
    WHERE lr.range_low IS NOT NULL
    OR hr.range_high IS NOT NULL
),
cohort_measurement_changes AS (
    SELECT example_id,
           person_id,
           measurement_concept_id,
           unit_source_value,
           concept_name,
           value_as_number - LAG(value_as_number) OVER (
               PARTITION BY example_id,
                            measurement_concept_id
               ORDER BY feature_start_date
           ) AS change,
           feature_start_date,
           person_end_date
    FROM cohort_measurements_with_values
),
cohort_quartiles AS (
    SELECT measurement_concept_id,
           unit_source_value,
           PERCENTILE_DISC(0.25) WITHIN GROUP (ORDER BY value_as_number) AS measurement_25,
           PERCENTILE_DISC(0.50) WITHIN GROUP (ORDER BY value_as_number) AS measurement_50,
           PERCENTILE_DISC(0.75) WITHIN GROUP (ORDER BY value_as_number) AS measurement_75
    FROM cohort_measurements_with_values
    GROUP BY measurement_concept_id,
             unit_source_value
),
cohort_measurements_with_values_and_quartiles AS (
    SELECT m.*,
           q.measurement_25,
           q.measurement_50,
           q.measurement_75
    FROM cohort_measurements_with_values m
    JOIN cohort_quartiles q
    ON m.measurement_concept_id = q.measurement_concept_id
    AND m.unit_source_value = q.unit_source_value
)
SELECT example_id,
       person_id,
       measurement_concept_id || ' - lab - ' || concept_name || ' - below 25th percentile' AS concept_name,
       feature_start_date,
       person_end_date
FROM cohort_measurements_with_values_and_quartiles
WHERE value_as_number < measurement_25 -- if 25th percentile is minimum, this quartile will be empty
UNION
SELECT example_id,
       person_id,
       measurement_concept_id || ' - lab - ' || concept_name || ' - 25th to 50th percentile' AS concept_name,
       feature_start_date,
       person_end_date
FROM cohort_measurements_with_values_and_quartiles
WHERE value_as_number >= measurement_25
AND value_as_number < measurement_50 -- if 25th and 50th percentiles identical, this quartile will be empty
UNION
SELECT example_id,
       person_id,
       measurement_concept_id || ' - lab - ' || concept_name || ' - 50th to 75th percentile' AS concept_name,
       feature_start_date,
       person_end_date
FROM cohort_measurements_with_values_and_quartiles
WHERE value_as_number >= measurement_50
AND value_as_number < measurement_75 -- if 50th and 75th percentiles identical, this quartile will be empty
UNION
SELECT example_id,
       person_id,
       measurement_concept_id || ' - lab - ' || concept_name || ' - above 75th percentile' AS concept_name,
       feature_start_date,
       person_end_date
FROM cohort_measurements_with_values_and_quartiles 
WHERE value_as_number >= measurement_75
UNION
SELECT example_id,
       person_id,
       measurement_concept_id || ' - lab - ' || concept_name || ' - increasing' AS concept_name,
       feature_start_date,
       person_end_date
FROM cohort_measurement_changes
WHERE change > 0
AND change IS NOT NULL
UNION
SELECT example_id,
       person_id,
       measurement_concept_id || ' - lab - ' || concept_name || ' - decreasing' AS concept_name,
       feature_start_date,
       person_end_date
FROM cohort_measurement_changes
WHERE change < 0
AND change IS NOT NULL
UNION
SELECT example_id,
       person_id,
       measurement_concept_id || ' - lab - ' || concept_name || ' - below range' AS concept_name,
       feature_start_date,
       person_end_date
FROM cohort_measurements_with_values_and_references
WHERE value_as_number < range_low
AND range_low IS NOT NULL
UNION
SELECT example_id,
       person_id,
       measurement_concept_id || ' - lab - ' || concept_name || ' - in range' AS concept_name,
       feature_start_date,
       person_end_date
FROM cohort_measurements_with_values_and_references
WHERE (value_as_number >= range_low
       AND value_as_number <= range_high
       AND range_low IS NOT NULL
       AND range_high IS NOT NULL
      )
OR (range_low IS NULL
    AND value_as_number <= range_high
    AND range_high IS NOT NULL
   )
OR (range_high IS NULL
    AND value_as_number >= range_low
    AND range_low IS NOT NULL
   )
UNION
SELECT example_id,
       person_id,
       measurement_concept_id || ' - lab - ' || concept_name || ' - above range' AS concept_name,
       feature_start_date,
       person_end_date
FROM cohort_measurements_with_values_and_references
WHERE value_as_number > range_high
AND range_high IS NOT NULL
UNION
SELECT example_id,
       person_id,
       measurement_concept_id || ' - lab - ' || concept_name || ' - ordered' AS concept_name,
       feature_start_date,
       person_end_date
FROM cohort_measurements_with_names