DROP TABLE IF EXISTS {schema_name}.omop_cohort_{window_name}{eol_suffix};
CREATE TABLE {schema_name}.omop_cohort_{window_name}{eol_suffix} AS 
WITH max_prediction_date AS (
    SELECT MAX(DATE(prediction_date)) AS prediction_date
    FROM {schema_name}.prediction_dates_{window_name}
),
expired_cohort AS (
   SELECT DISTINCT person_id,
          outcome_date,
          1 as y
   FROM {schema_name}.monthly_eligibility_{window_name}{eol_suffix}
   WHERE y = 1
),
not_expired_cohort AS ( 
   SELECT DISTINCT me.person_id,
          CAST(NULL AS DATE) AS outcome_date,
          0 as y
   FROM {schema_name}.monthly_eligibility_{window_name}{eol_suffix} me
   WHERE NOT EXISTS (
       SELECT
       FROM expired_cohort ec
       WHERE ec.person_id = me.person_id
    )
),
combined_cohort AS (
    SELECT ec.person_id,
           d.prediction_date AS end_date,
           ec.outcome_date,
           ec.y
    FROM expired_cohort ec
    CROSS JOIN max_prediction_date d
    UNION
    SELECT nec.person_id,
           d.prediction_date AS end_date,
           nec.outcome_date,
           nec.y
    FROM not_expired_cohort nec
    CROSS JOIN max_prediction_date d
)
SELECT ROW_NUMBER() OVER (ORDER BY person_id) - 1 AS example_id,
       person_id,
       end_date,
       outcome_date,
       y
FROM combined_cohort
ORDER BY person_id,
         end_date;

CREATE INDEX idx_{schema_name}_omop_cohort_{window_name}{eol_suffix}_example_id
ON {schema_name}.omop_cohort_{window_name}{eol_suffix} (
    example_id ASC
);

CREATE INDEX idx_{schema_name}_omop_cohort_{window_name}{eol_suffix}
ON {schema_name}.omop_cohort_{window_name}{eol_suffix} (
    person_id ASC,
    end_date ASC
);