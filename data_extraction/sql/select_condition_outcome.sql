DROP TABLE IF EXISTS {schema_name}.condition_{outcome_id}_outcomes{debug_suffix};
CREATE TABLE {schema_name}.condition_{outcome_id}_outcomes{debug_suffix} AS
WITH condition_concept_occurrences AS (
    SELECT DISTINCT co.person_id,
           co.condition_start_date
    FROM cdm.condition_occurrence co
    WHERE co.condition_concept_id = {outcome_id}
),
cohort_ids AS (
    SELECT DISTINCT person_id
    FROM {schema_name}.omop_cohort_{window_name}{debug_suffix}
),
cohort_end_dates AS (
    SELECT DISTINCT me.person_id,
           me.end_date
    FROM cohort_ids c
    JOIN {schema_name}.monthly_eligibility_{window_name} me
    ON c.person_id = me.person_id
),
cohort_concept_occurrences AS (
    SELECT me.person_id,
           me.end_date,
           MIN(co.condition_start_date) AS outcome_date,
           1 AS y
    FROM cohort_end_dates me
    JOIN condition_concept_occurrences co
    ON me.person_id = co.person_id
    AND me.end_date < co.condition_start_date -- features from end_date are used, so excluded from outcome period
    AND co.condition_start_date <= me.end_date + 3 * interval '1 month'
    GROUP BY me.person_id,
             me.end_date
),
-- for condition outcomes, exclude patients who have previously had the condition
-- so that model can focus on diagnosis
-- may remove this criteria for non-chronic conditions
first_concept_occurrence AS (
    SELECT c.person_id, 
           MIN(co.condition_start_date) AS first_occurrence_date
    FROM cohort_ids c
    JOIN condition_concept_occurrences co
    ON c.person_id = co.person_id
    GROUP BY c.person_id
)
SELECT ROW_NUMBER() OVER (ORDER BY me.person_id, me.end_date) - 1 AS example_id,
       me.person_id,
       me.end_date,
       co.outcome_date,
       COALESCE(co.y, 0) AS y
FROM cohort_end_dates me
LEFT JOIN cohort_concept_occurrences co
ON me.person_id = co.person_id
AND me.end_date = co.end_date
WHERE NOT EXISTS (
    SELECT
    FROM first_concept_occurrence fo
    WHERE me.person_id = fo.person_id
    AND fo.first_occurrence_date <= me.end_date
)
ORDER BY me.person_id,
         me.end_date;