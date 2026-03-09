import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('postgresql://postgres:admin@localhost:5433/aquaformer')

sql_query = """SELECT reading_timestamp,temp_celsius,actual_precip_mm FROM climate_data WHERE latitude = 47.5  AND longitude = -122.25 ORDER BY reading_timestamp LIMIT 24 """

df_seattle = pd.read_sql(sql_query , engine)

print(df_seattle)