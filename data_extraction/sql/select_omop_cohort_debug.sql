DROP TABLE IF EXISTS {schema_name}.omop_cohort_{window_name}{eol_suffix}{debug_suffix};
CREATE TABLE {schema_name}.omop_cohort_{window_name}{eol_suffix}{debug_suffix} AS 
WITH random_subset AS (
    SELECT * 
    FROM {schema_name}.omop_cohort_{window_name}{eol_suffix}
    ORDER BY RANDOM()
    LIMIT {debug_size}
)
SELECT *
FROM random_subset
ORDER BY person_id,
         end_date;

CREATE INDEX idx_{schema_name}_omop_cohort_{window_name}{eol_suffix}{debug_suffix}_example_id
ON {schema_name}.omop_cohort_{window_name}{eol_suffix}{debug_suffix} (
    example_id ASC
);

CREATE INDEX idx_{schema_name}_omop_cohort_{window_name}{eol_suffix}{debug_suffix}
ON {schema_name}.omop_cohort_{window_name}{eol_suffix}{debug_suffix} (
    person_id ASC,
    end_date ASC
);