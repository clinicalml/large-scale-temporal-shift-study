-- Override references with manually standardized references where available
DROP TABLE IF EXISTS {measurement_aux_schema}.measurement_age_gender_specific_standardized_{direction}_references;
CREATE TABLE {measurement_aux_schema}.measurement_age_gender_specific_standardized_{direction}_references AS
SELECT r.concept_id,
       r.unit_source_value,
       r.concept_name,
       r.age_range,
       r.gender_source_value,
       r.similar_concept_id,
       r.similar_concept_name,
       COALESCE(us.male_range_{direction}, s.male_range_{direction}, r.range_{direction}) AS range_{direction},
       r.from_opposite_gender,
       r.from_different_age_range,
       r.from_general_reference
FROM {measurement_aux_schema}.measurement_age_gender_specific_{direction}_references r
LEFT JOIN {measurement_aux_schema}.measurement_unit_specific_references_from_sources us
ON r.concept_id = us.concept_id
AND r.unit_source_value = us.unit_source_value
LEFT JOIN {measurement_aux_schema}.measurement_references_from_sources s
ON r.concept_id = s.concept_id
WHERE r.gender_source_value = 'M'
UNION
SELECT r.concept_id,
       r.unit_source_value,
       r.concept_name,
       r.age_range,
       r.gender_source_value,
       r.similar_concept_id,
       r.similar_concept_name,
       COALESCE(us.female_range_{direction}, s.female_range_{direction}, r.range_{direction}) AS range_{direction},
       r.from_opposite_gender,
       r.from_different_age_range,
       r.from_general_reference
FROM {measurement_aux_schema}.measurement_age_gender_specific_{direction}_references r
LEFT JOIN {measurement_aux_schema}.measurement_unit_specific_references_from_sources us
ON r.concept_id = us.concept_id
AND r.unit_source_value = us.unit_source_value
LEFT JOIN {measurement_aux_schema}.measurement_references_from_sources s
ON r.concept_id = s.concept_id
WHERE r.gender_source_value = 'F';

CREATE INDEX idx_{measurement_aux_schema}_age_gender_standardized_{direction}_references
ON {measurement_aux_schema}.measurement_age_gender_specific_standardized_{direction}_references (
    concept_id ASC,
    unit_source_value ASC,
    age_range ASC,
    gender_source_value ASC
);