DROP TABLE IF EXISTS {schema_name}.monthly_eligibility_{window_name};
CREATE TABLE {schema_name}.monthly_eligibility_{window_name} AS
-- select patients who are observed on prediction date
WITH obs_patients_on_date AS (
    SELECT DISTINCT o.person_id,
           p.prediction_date
    FROM cdm.observation_period o
    JOIN {schema_name}.prediction_dates_{window_name} p
    ON o.observation_period_start_date <= p.prediction_date
    AND o.observation_period_end_date >= p.prediction_date
),
obs_patients_on_date_with_death_date AS (
    SELECT o.person_id,
           o.prediction_date,
           p.death_datetime AS death_date
    FROM obs_patients_on_date o
    JOIN cdm.person p
    ON o.person_id = p.person_id
),
-- select patients who expired within 3 months
-- keep death_date as outcome_date since just used for omop cohort part anyways
expired_outcome_labels AS (
    SELECT DISTINCT person_id,
           prediction_date AS end_date,
           death_date AS outcome_date,
           1 as y
    FROM obs_patients_on_date_with_death_date 
    WHERE death_date IS NOT NULL
    AND death_date <= prediction_date + INTERVAL '3 months'
),
-- select patients who did not expire within 3 months
not_expired_outcome_labels AS (
    SELECT DISTINCT person_id,
           prediction_date AS end_date,
           CAST(NULL AS DATE) AS outcome_date,
           0 as y
    FROM obs_patients_on_date_with_death_date 
    WHERE (death_date IS NULL
           OR death_date > prediction_date + INTERVAL '3 months')
),
-- check patients who did not expire are observed some time 3 months after prediction date
not_expired_observed_outcome_labels AS (
    SELECT DISTINCT p.person_id,
           p.end_date,
           p.outcome_date,
           p.y
    FROM not_expired_outcome_labels p
    JOIN cdm.observation_period o
    ON p.person_id = o.person_id
    AND o.observation_period_end_date >= p.end_date + INTERVAL '3 months'
),
-- union of patients who expired within 3 months and patients who did not expire and are observed some time 3 months after
union_cohort AS (
    SELECT person_id,
           end_date,
           outcome_date,
           y
    FROM expired_outcome_labels
    UNION
    SELECT person_id,
           end_date,
           outcome_date,
           y
    FROM not_expired_observed_outcome_labels
),
-- select patients who are observed for at least 95% of the window before prediction date - hardest to compute so do last
cohort_obs_length AS (
    SELECT p.person_id,
           p.end_date,
           p.outcome_date,
           p.y,
           SUM(GREATEST(EXTRACT(EPOCH FROM (LEAST(o.observation_period_end_date, p.end_date)
                                            - GREATEST(o.observation_period_start_date, 
                                                       p.end_date - INTERVAL '{window}')
                                           )
                               )/(24*60*60), 
                        0)
              ) AS num_days
    FROM union_cohort p
    JOIN cdm.observation_period o
    ON p.person_id = o.person_id
    AND p.end_date >= o.observation_period_start_date
    AND p.end_date <= o.observation_period_end_date
    GROUP BY p.person_id,
             p.end_date,
             p.outcome_date,
             p.y
)
SELECT person_id,
       end_date,
       outcome_date,
       y,
       num_days
FROM cohort_obs_length
WHERE num_days > 0.95 * EXTRACT(EPOCH FROM (INTERVAL '{window}'))/(24*60*60)
ORDER BY person_id,
         end_date;

CREATE INDEX idx_{schema_name}_monthly_eligibility_{window_name}
ON {schema_name}.monthly_eligibility_{window_name} (
    person_id ASC,
    end_date ASC
);