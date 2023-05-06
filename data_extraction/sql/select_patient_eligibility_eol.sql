DROP TABLE IF EXISTS {schema_name}.monthly_eligibility_{window_name}_eol;
CREATE TABLE {schema_name}.monthly_eligibility_{window_name}_eol AS
-- filter people by min age on last prediction date
WITH max_pred_date AS (
    SELECT MAX(DATE(prediction_date)) as prediction_date
    FROM {schema_name}.prediction_dates_{window_name}
),
age_people_any_date AS (
    SELECT DISTINCT p.person_id,
           p.year_of_birth,
           DATE(p.death_datetime) AS death_date
    FROM cdm.person p
    CROSS JOIN max_pred_date d
    WHERE EXTRACT(YEAR FROM d.prediction_date) - p.year_of_birth > 70
    AND p.person_id > 0
),
-- select Medicare Advantage concepts
medicare_concepts AS (
    SELECT DISTINCT c.concept_id
    FROM cdm.concept c
    WHERE (c.concept_name = 'Medicare Health Maintenance Organization (HMO)' 
           OR c.concept_name = 'Medicare Preferred Provider Organization (PPO)')
),
-- filter patient on Medicare Advantage at any time
medicare_patients AS (
    SELECT DISTINCT ap.person_id,
           ap.year_of_birth,
           ap.death_date,
           ppp.payer_plan_period_start_date,
           ppp.payer_plan_period_end_date
    FROM age_people_any_date ap
    JOIN cdm.payer_plan_period ppp
    ON ap.person_id = ppp.person_id
    JOIN cdm.concept c
    ON ppp.payer_concept_id = c.concept_id
),
-- select observation periods for people who are in cohort
cohort_observation_periods AS (
    SELECT DISTINCT p.person_id,
           o.observation_period_start_date,
           o.observation_period_end_date
    FROM medicare_patients p
    JOIN cdm.observation_period o
    ON p.person_id = o.person_id
),
-- now cross join with prediction date to filter age and Medicare Advantage on prediction date
-- select patients who are not expired before prediction date
medicare_patients_on_date AS (
    SELECT DISTINCT p.person_id,
           DATE(d.prediction_date) AS prediction_date,
           p.death_date
    FROM medicare_patients p
    CROSS JOIN {schema_name}.prediction_dates_{window_name} d
    WHERE EXTRACT(YEAR FROM DATE(d.prediction_date)) - p.year_of_birth > 70
    AND (p.death_date IS NULL 
         OR DATE(d.prediction_date) <= p.death_date)
    AND p.payer_plan_period_start_date <= DATE(d.prediction_date)
    AND p.payer_plan_period_end_date >= DATE(d.prediction_date)
),
-- select patients who are observed on prediction date
obs_patients_on_date AS (
    SELECT DISTINCT p.person_id,
           p.prediction_date,
           p.death_date
    FROM medicare_patients_on_date p
    JOIN cohort_observation_periods o
    ON o.person_id = p.person_id
    AND o.observation_period_start_date <= p.prediction_date
    AND o.observation_period_end_date >= p.prediction_date
),
-- select patients who expired within 3 months
expired_outcome_labels AS (
    SELECT DISTINCT person_id,
           prediction_date AS end_date,
           death_date AS outcome_date,
           1 as y
    FROM obs_patients_on_date
    WHERE death_date IS NOT NULL
    AND death_date <= prediction_date + INTERVAL '3 months'
),
-- select patients who did not expire within 3 months
not_expired_outcome_labels AS (
    SELECT DISTINCT person_id,
           prediction_date AS end_date,
           CAST(NULL AS DATE) AS outcome_date,
           0 as y
    FROM obs_patients_on_date 
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
    JOIN cohort_observation_periods o
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
    JOIN cohort_observation_periods o
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

CREATE INDEX idx_{schema_name}_monthly_eligibility_{window_name}_eol
ON {schema_name}.monthly_eligibility_{window_name}_eol (
    person_id ASC,
    end_date ASC
);