WITH cohort_first_outcome_occurrences AS ( 
    SELECT DISTINCT p.person_id, 
           o.condition_concept_id AS concept_id, 
           MIN(o.condition_start_date) AS first_condition_date 
    FROM {schema_name}.monthly_eligibility_3_years p 
    JOIN cdm.condition_occurrence o 
    ON p.person_id = o.person_id 
    GROUP BY p.person_id, 
             o.condition_concept_id 
), 
outcome_counts AS (
    SELECT o.concept_id, 
           COUNT(DISTINCT o.person_id) AS concept_count 
    FROM cohort_first_outcome_occurrences o 
    JOIN {schema_name}.monthly_eligibility_3_years p 
    ON o.person_id = p.person_id 
    AND p.end_date <= o.first_condition_date 
    AND p.end_date + 3 * INTERVAL '1 month' >= o.first_condition_date 
    GROUP BY o.concept_id 
) 
SELECT oc.concept_id, 
       c.concept_name 
FROM outcome_counts oc 
JOIN cdm.concept c 
ON oc.concept_id = c.concept_id
ORDER BY oc.concept_count DESC,
         oc.concept_id ASC
LIMIT {number_outcomes};