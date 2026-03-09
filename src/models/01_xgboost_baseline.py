import pandas as pd
import numpy as np
import xgboost as xgb
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# STEP 1: CONNECT TO DATABASE & PULL DATA
engine = create_engine('postgresql://postgres:admin@localhost:5433/aquaformer')

# Pulled in the missing columns (pressure and wind) so X has what it needs!
query = """
    SELECT reading_timestamp, temp_celsius, pressure_hpa, wind_u_vector, wind_v_vector, actual_precip_mm 
    FROM climate_data 
    WHERE latitude = 47.5 AND longitude = -122.25 
    ORDER BY reading_timestamp
"""
df = pd.read_sql(query, engine)

# Drop any rows with missing data just to be safe so XGBoost doesn't crash
df = df.dropna()

# STEP 2: DEFINE THE EXTREME EVENT TARGET
# Finding the 99th percentile - the literal top 1% of storms
threshold = df['actual_precip_mm'].quantile(0.99)

# Creating the target column and forcing it to be 1s and 0s instead of True/False
df['is_extreme'] = (df['actual_precip_mm'] >= threshold).astype(int)

# STEP 3: PREPARE FEATURES (X) AND TARGET (y)
# Grabbing only the weather variables
# Left actual_precip_mm out so the model can't cheat by looking at the answer!
X = df[['temp_celsius', 'pressure_hpa', 'wind_u_vector', 'wind_v_vector']]

# Swapped this to a lowercase 'y' to match the train_test_split below
y = df['is_extreme'] 

# STEP 4: TRAIN-TEST SPLIT
# Standard 80/20 split for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 5: HANDLE CLASS IMBALANCE
# Calculating the penalty for missing a flood (should be around 99)
ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

# STEP 6: TRAIN THE XGBOOST MODEL
model = xgb.XGBClassifier(
    scale_pos_weight=ratio,
    eval_metric='logloss',
    use_label_encoder=False
)
model.fit(X_train, y_train)


# STEP 7: EVALUATE PERFORMANCE
predictions = model.predict(X_test)
print(classification_report(y_test, predictions, target_names=['Normal (0)', 'Extreme (1)']))
print(confusion_matrix(y_test, predictions))
