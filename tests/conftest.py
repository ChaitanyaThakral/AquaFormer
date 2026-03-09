import pytest
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


@pytest.fixture(scope="session")
def db_engine():
    """Shared PostgreSQL engine for live-DB validation tests."""
    engine = create_engine('postgresql://postgres:admin@localhost:5433/aquaformer')
    yield engine
    engine.dispose()


@pytest.fixture
def dummy_weather_df():
    """99 hours of normal weather + 1 hour of extreme rainfall."""
    np.random.seed(42)
    n = 100
    data = {
        'temp_celsius': np.random.uniform(5, 25, n),
        'pressure_hpa': np.random.uniform(1000, 1020, n),
        'wind_u_vector': np.random.uniform(-5, 5, n),
        'wind_v_vector': np.random.uniform(-5, 5, n),
        'actual_precip_mm': list(np.random.uniform(0, 5, 99)) + [150.0],
    }
    return pd.DataFrame(data)
