import pandas as pd
import numpy as np
import xgboost as xgb
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def compute_extreme_threshold(df: pd.DataFrame, quantile: float = 0.99) -> float:
    """Return the precipitation value at the given quantile."""
    return float(df['actual_precip_mm'].quantile(quantile))


def create_target_column(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Add a binary 'is_extreme' column: 1 if precip >= threshold, else 0."""
    df = df.copy()
    df['is_extreme'] = (df['actual_precip_mm'] >= threshold).astype(int)
    return df


def prepare_features(df: pd.DataFrame):
    """Split into feature matrix X and target vector y.

    X contains only weather variables (no precipitation or target leakage).
    """
    feature_cols = ['temp_celsius', 'pressure_hpa', 'wind_u_vector', 'wind_v_vector']
    X = df[feature_cols]
    y = df['is_extreme']
    return X, y


def compute_class_weight(y_train: pd.Series) -> float:
    """Calculate scale_pos_weight for imbalanced classification."""
    return float(np.sum(y_train == 0)) / np.sum(y_train == 1)


def train_model(X_train: pd.DataFrame, y_train: pd.Series, scale_pos_weight: float):
    """Train and return an XGBClassifier."""
    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
    )
    model.fit(X_train, y_train)
    return model


if __name__ == '__main__':
    # STEP 1: CONNECT TO DATABASE & PULL DATA
    engine = create_engine('postgresql://postgres:admin@localhost:5433/aquaformer')

    query = """
        SELECT reading_timestamp, temp_celsius, pressure_hpa, wind_u_vector, wind_v_vector, actual_precip_mm
        FROM climate_data
        WHERE latitude = 47.5 AND longitude = -122.25
        ORDER BY reading_timestamp
    """
    df = pd.read_sql(query, engine)
    df = df.dropna()

    # STEP 2: DEFINE THE EXTREME EVENT TARGET
    threshold = compute_extreme_threshold(df)
    df = create_target_column(df, threshold)

    # STEP 3: PREPARE FEATURES AND TARGET
    X, y = prepare_features(df)

    # STEP 4: TRAIN-TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # STEP 5: HANDLE CLASS IMBALANCE
    ratio = compute_class_weight(y_train)

    # STEP 6: TRAIN THE XGBOOST MODEL
    model = train_model(X_train, y_train, ratio)

    # STEP 7: EVALUATE PERFORMANCE
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions, target_names=['Normal (0)', 'Extreme (1)']))
    print(confusion_matrix(y_test, predictions))
