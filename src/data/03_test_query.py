import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('postgresql://postgres:admin@localhost:5435/aquaformer')

sql_query = """SELECT * From climate_data LIMIT 24 """

df_seattle = pd.read_sql(sql_query , engine)

print(df_seattle)