SELECT *
FROM {schema_name}.omop_cohort_{window_name}{eol_suffix}
ORDER BY person_id,
         end_date;