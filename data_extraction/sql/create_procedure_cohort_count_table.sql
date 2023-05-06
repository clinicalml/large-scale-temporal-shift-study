CREATE TABLE {schema_name}.cohort_procedure_outcome_counts AS 
WITH cohort_outcome_occurrences AS ( 
    SELECT DISTINCT c.person_id, 
           c.end_date, 
           o.procedure_concept_id AS concept_id 
    FROM {schema_name}.monthly_eligibility_1_year c 
    JOIN cdm.procedure_occurrence o 
    ON c.person_id = o.person_id 
    AND c.end_date <= o.procedure_date 
    AND c.end_date + 3 * INTERVAL '1 month' >= o.procedure_date 
), 
cohort_outcome_counts AS ( 
    SELECT concept_id, 
           COUNT(*) AS concept_count 
    FROM cohort_outcome_occurrences 
    GROUP BY concept_id 
) 
SELECT po.concept_id, 
       c.concept_name, 
       po.concept_count 
FROM cohort_outcome_counts po 
JOIN cdm.concept c 
ON po.concept_id = c.concept_id 
ORDER BY po.concept_count DESC;