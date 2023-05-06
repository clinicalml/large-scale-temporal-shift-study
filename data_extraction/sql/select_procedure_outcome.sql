DROP TABLE IF EXISTS {schema_name}.procedure_{outcome_name}_outcomes{debug_suffix};
CREATE TABLE {schema_name}.procedure_{outcome_name}_outcomes{debug_suffix} AS
WITH procedure_concept_occurrences AS (
    SELECT DISTINCT po.person_id,
           po.procedure_date
    FROM cdm.procedure_occurrence po
    WHERE po.procedure_concept_id IN ({outcome_id})
),
cohort_end_dates AS (
    SELECT DISTINCT me.person_id,
           me.end_date
    FROM {schema_name}.omop_cohort_{window_name}{debug_suffix} c
    JOIN {schema_name}.monthly_eligibility_{window_name} me
    ON c.person_id = me.person_id
),
cohort_concept_occurrences AS (
    SELECT me.person_id,
           me.end_date,
           MIN(co.procedure_date) AS outcome_date,
           1 AS y
    FROM cohort_end_dates me
    JOIN procedure_concept_occurrences co
    ON me.person_id = co.person_id
    AND me.end_date < co.procedure_date -- features from end_date are used, so excluded from outcome period
    AND co.procedure_date <= me.end_date + 3 * interval '1 month'
    GROUP BY me.person_id,
             me.end_date
)
-- for procedure outcomes, prior procedures are okay because when a procedure is repeated can be interesting
SELECT ROW_NUMBER() OVER (ORDER BY me.person_id, me.end_date) - 1 AS example_id,
       me.person_id,
       me.end_date,
       co.outcome_date,
       COALESCE(co.y, 0) AS y
FROM cohort_end_dates me
LEFT JOIN cohort_concept_occurrences co
ON me.person_id = co.person_id
AND me.end_date = co.end_date
ORDER BY me.person_id,
         me.end_date;