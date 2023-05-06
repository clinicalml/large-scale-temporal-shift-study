/*********************************************************************************
Script to create indexes and constraints on cdm_measurement_aux tables
********************************************************************************/

SET SEARCH_PATH TO cdm_measurement_aux;

/************************
Primary key constraints
************************/

ALTER TABLE measurement_with_nulls_replacing_zero 
ADD CONSTRAINT xpk_measurement_with_nulls_replacing_zero 
PRIMARY KEY (measurement_id);


/************************
Indices
************************/

CREATE INDEX idx_measurement_with_nulls_replacing_zero_person_id 
ON measurement_with_nulls_replacing_zero (person_id ASC);

CLUSTER measurement_with_nulls_replacing_zero 
USING idx_measurement_with_nulls_replacing_zero_person_id;

CREATE INDEX idx_measurement_with_nulls_replacing_zero_concept_id 
ON measurement_with_nulls_replacing_zero (measurement_concept_id ASC);

CREATE INDEX idx_measurement_with_nulls_replacing_zero_visit_id 
ON measurement_with_nulls_replacing_zero (visit_occurrence_id ASC);


/************************
Custom indices
************************/

CREATE INDEX idx_measurement_with_nulls_replacing_zero_date 
ON measurement_with_nulls_replacing_zero (measurement_date ASC);

CREATE INDEX idx_measurement_with_nulls_replacing_zero_person_concept 
ON measurement_with_nulls_replacing_zero (person_id ASC, measurement_concept_id ASC);

CREATE INDEX idx_measurement_with_nulls_replacing_zero_person_concept_date 
ON measurement_with_nulls_replacing_zero (person_id ASC, measurement_date ASC);

CREATE INDEX idx_measurement_with_nulls_replacing_zero_concept_unit
ON measurement_with_nulls_replacing_zero (measurement_concept_id ASC, unit_source_value ASC);

CREATE INDEX idx_measurement_with_nulls_replacing_zero_concept_value
ON measurement_with_nulls_replacing_zero (measurement_concept_id ASC, value_as_number ASC);

CREATE INDEX idx_measurement_with_nulls_replacing_zero_concept_unit_value
ON measurement_with_nulls_replacing_zero (measurement_concept_id ASC, unit_source_value ASC, value_as_number ASC);


/************************
Indices for reference tables
************************/

CREATE INDEX idx_measurement_high_references_concept_id
ON measurement_high_references (concept_id ASC);

CREATE INDEX idx_measurement_low_references_concept_id
ON measurement_low_references (concept_id ASC);