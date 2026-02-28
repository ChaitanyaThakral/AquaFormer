CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE SpatialGrid (
    grid_id SERIAL PRIMARY KEY,
    location GEOMETRY(Point, 4326) NOT NULL, 
    elevation_meters NUMERIC(7, 2),
    region_type VARCHAR(50) CHECK (region_type IN ('Coastal', 'Mountainous', 'Urban', 'Valley'))
);

CREATE INDEX idx_spatialgrid_location ON SpatialGrid USING GIST (location);

CREATE TABLE ClimateReadings (
    reading_id BIGSERIAL PRIMARY KEY,
    grid_id INTEGER REFERENCES SpatialGrid(grid_id) ON DELETE CASCADE,
    reading_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    temp_celsius NUMERIC(5, 2),
    pressure_hpa NUMERIC(6, 2),
    wind_u_vector NUMERIC(5, 2), 
    wind_v_vector NUMERIC(5, 2), 
    specific_humidity NUMERIC(6, 4),
    actual_precip_mm NUMERIC(6, 2), 
    is_imputed BOOLEAN DEFAULT FALSE,
    bayesian_uncertainty_bounds NUMERIC(5, 4), 
    UNIQUE(grid_id, reading_timestamp) 
);

CREATE INDEX idx_climate_grid_time ON ClimateReadings (grid_id, reading_timestamp DESC);
