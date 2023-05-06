SET SEARCH_PATH TO {schema_name};

-- create table for general references
DROP TABLE IF EXISTS measurement_references_from_sources;
CREATE TABLE measurement_references_from_sources (
    concept_id INTEGER NOT NULL,
    male_range_low NUMERIC NULL,
    male_range_high NUMERIC NULL,
    female_range_low NUMERIC NULL,
    female_range_high NUMERIC NULL
);

-- create table for unit specific references
DROP TABLE IF EXISTS measurement_unit_specific_references_from_sources;
CREATE TABLE measurement_unit_specific_references_from_sources (
    concept_id INTEGER NOT NULL,
    unit_source_value VARCHAR(50) NOT NULL,
    male_range_low NUMERIC NULL,
    male_range_high NUMERIC NULL,
    female_range_low NUMERIC NULL,
    female_range_high NUMERIC NULL
);

-- create table for units to drop
DROP TABLE IF EXISTS measurement_units_to_drop;
CREATE TABLE measurement_units_to_drop (
    concept_id INTEGER NOT NULL,
    unit_source_value VARCHAR(50) NOT NULL
);

-- create table for out of range definitions
DROP TABLE IF EXISTS measurements_out_of_range;
CREATE TABLE measurements_out_of_range (
    concept_id INTEGER NOT NULL,
    unit_source_value VARCHAR(50) NOT NULL,
    lower_bound NUMERIC NULL,
    upper_bound NUMERIC NULL
);