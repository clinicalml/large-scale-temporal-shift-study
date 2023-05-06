DROP TABLE IF EXISTS {schema_name}.lab_group_{outcome_name}_{direction}_outcomes{debug_suffix};
CREATE TABLE {schema_name}.lab_group_{outcome_name}_{direction}_outcomes{debug_suffix} AS
WITH cohort_ids AS (
    SELECT DISTINCT person_id
    FROM {schema_name}.omop_cohort_{window_name}{debug_suffix}
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
    SELECT c.person_id,
           EXTRACT(YEAR FROM m.measurement_date) - c.year_of_birth AS age,
           c.gender_source_value,
           m.measurement_concept_id AS concept_id,
           m.value_as_number,
           m.unit_source_value,
           m.measurement_date
    FROM cohort_birth_year_genders c
    JOIN {schema_name}.measurement_with_nulls_replacing_zero_drop_nonstandard m
    ON c.person_id = m.person_id
    WHERE m.measurement_concept_id IN ({outcome_id})
    AND m.value_as_number IS NOT NULL
),
cohort_outcome_measurements_age_range AS (
    SELECT person_id,
           CASE WHEN age <= 30
                THEN 'range 1: <= 30'
                WHEN age <= 50
                THEN 'range 2: 31 - 50'
                WHEN age <= 70
                THEN 'range 3: 51 - 70'
                ELSE 'range 4: > 70'
           END AS age_range,
           gender_source_value,
           concept_id,
           value_as_number,
           unit_source_value,
           measurement_date
    FROM cohort_outcome_measurements
),
outcome_references AS (
    SELECT age_range,
           gender_source_value,
           concept_id,
           unit_source_value,
           range_{direction}
    FROM {schema_name}.measurement_age_gender_specific_standardized_{direction}_references
    WHERE concept_id IN ({outcome_id})
),
cohort_abnormal_measurements AS (
    SELECT c.person_id,
           c.measurement_date
    FROM cohort_outcome_measurements_age_range c
    JOIN outcome_references r
    ON c.age_range = r.age_range
    AND c.gender_source_value = r.gender_source_value
    AND c.concept_id = r.concept_id
    AND c.unit_source_value = r.unit_source_value
    WHERE c.value_as_number {sign} range_{direction}
),
cohort_end_dates AS (
    SELECT DISTINCT me.person_id,
           me.end_date
    FROM cohort_ids c
    JOIN {schema_name}.monthly_eligibility_{window_name} me
    ON c.person_id = me.person_id
),
cohort_abnormal_outcomes AS (
    SELECT c.person_id,
           c.end_date,
           MIN(m.measurement_date) AS outcome_date,
           1 AS y
    FROM cohort_end_dates c
    JOIN cohort_abnormal_measurements m
    ON c.person_id = m.person_id
    AND c.end_date < m.measurement_date -- features from end_date are used, so excluded from outcome period
    AND m.measurement_date <= c.end_date + 3 * interval '1 month'
    GROUP BY c.person_id,
             c.end_date
)
-- for lab outcomes, prior abnormal labs are okay because when a measurement is repeated can be interesting
SELECT ROW_NUMBER() OVER (ORDER BY me.person_id, me.end_date) - 1 AS example_id,
       me.person_id,
       me.end_date,
       co.outcome_date,
       COALESCE(co.y, 0) AS y
FROM cohort_end_dates me
LEFT JOIN cohort_abnormal_outcomes co
ON me.person_id = co.person_id
AND me.end_date = co.end_date
ORDER BY me.person_id,
         me.end_date;