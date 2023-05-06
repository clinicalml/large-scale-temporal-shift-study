-- Drop units that do not make sense and measurements that are out of range
DROP TABLE IF EXISTS {schema_name}.measurement_with_nulls_replacing_zero_drop_nonstandard;
CREATE TABLE {schema_name}.measurement_with_nulls_replacing_zero_drop_nonstandard AS
SELECT m.*
FROM cdm_measurement_aux.measurement_with_nulls_replacing_zero m
WHERE NOT EXISTS (
    SELECT
    FROM {schema_name}.measurement_units_to_drop d
    WHERE m.measurement_concept_id = d.concept_id
    AND m.unit_source_value = d.unit_source_value
)
AND NOT EXISTS (
    SELECT
    FROM {schema_name}.measurements_out_of_range r
    WHERE m.measurement_concept_id = r.concept_id
    AND m.unit_source_value = r.unit_source_value
    AND m.value_as_number IS NOT NULL
    AND ((r.upper_bound IS NOT NULL
         AND m.value_as_number > r.upper_bound)
        OR (r.lower_bound IS NOT NULL
            AND m.value_as_number < r.lower_bound))
);

-- Add same indices as measurement table
SET SEARCH_PATH TO {schema_name};

/************************
Primary key constraints
************************/

ALTER TABLE measurement_with_nulls_replacing_zero_drop_nonstandard 
ADD CONSTRAINT xpk_measurement_with_nulls_replacing_zero_drop_nonstandard
PRIMARY KEY (measurement_id);


/************************
Indices
************************/

CREATE INDEX idx_measurement_drop_nonstandard_person_id 
ON measurement_with_nulls_replacing_zero_drop_nonstandard (person_id ASC);

CLUSTER measurement_with_nulls_replacing_zero_drop_nonstandard 
USING idx_measurement_drop_nonstandard_person_id;

CREATE INDEX idx_measurement_drop_nonstandard_concept_id 
ON measurement_with_nulls_replacing_zero_drop_nonstandard (measurement_concept_id ASC);

CREATE INDEX idx_measurement_drop_nonstandard_visit_id 
ON measurement_with_nulls_replacing_zero_drop_nonstandard (visit_occurrence_id ASC);


/************************
Custom indices
************************/

CREATE INDEX idx_measurement_drop_nonstandard_date 
ON measurement_with_nulls_replacing_zero_drop_nonstandard (measurement_date ASC);

CREATE INDEX idx_measurement_drop_nonstandard_person_concept 
ON measurement_with_nulls_replacing_zero_drop_nonstandard (person_id ASC, measurement_concept_id ASC);

CREATE INDEX idx_measurement_drop_nonstandard_person_concept_date 
ON measurement_with_nulls_replacing_zero_drop_nonstandard (person_id ASC, measurement_date ASC);

CREATE INDEX idx_measurement_drop_nonstandard_concept_unit
ON measurement_with_nulls_replacing_zero_drop_nonstandard (measurement_concept_id ASC, unit_source_value ASC);

CREATE INDEX idx_measurement_drop_nonstandard_concept_value
ON measurement_with_nulls_replacing_zero_drop_nonstandard (measurement_concept_id ASC, value_as_number ASC);

CREATE INDEX idx_measurement_drop_nonstandard_concept_unit_value
ON measurement_with_nulls_replacing_zero_drop_nonstandard (measurement_concept_id ASC, unit_source_value ASC, value_as_number ASC);