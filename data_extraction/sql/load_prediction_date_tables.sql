SET SEARCH_PATH TO {schema_name};

-- load general references
DROP TABLE IF EXISTS prediction_dates_1_year;
CREATE TABLE prediction_dates_1_year (
    prediction_date VARCHAR(50) NOT NULL
);

\copy prediction_dates_1_year FROM 'prediction_dates.txt';

DROP TABLE IF EXISTS prediction_dates_3_years;
CREATE TABLE prediction_dates_3_years AS
SELECT prediction_date
FROM prediction_dates_1_year
WHERE DATE(prediction_date) >= DATE('2016-01-01');