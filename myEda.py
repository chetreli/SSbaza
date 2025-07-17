import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

path = "C:\Games\SSbaza\electric_vehicles_spec_2025.csv"
df = pd.read_csv(path)
columns = pd.Series(df.columns)
df['model'] = df['model'].astype('string')
df['brand'] = df['brand'].astype('string')
df['model'] = df['model'].fillna('unknown')
cat_feat = ['brand', 'model', 'battery_type', 'fast_charge_port', 'cargo_volume_l', 'drivetrain', 'segment', 'car_body_type', 'source_url']
num_feat = df.select_dtypes(include=['number']).columns
extra_df = df[num_feat]
print(df.info())
# with pd.option_context('display.max_columns', None):
#     print(df.describe())

#Plot that show how top speed depends on the number of seats
# df_show = extra_df.groupby('seats').mean()
#sns.lineplot(df_show['top_speed_kmh'])


# Plot of dependency the Top speed with battery capacity with taking into account the body type
# g = sns.FacetGrid(df, col = 'drivetrain', row='car_body_type')
# g.map(sns.scatterplot, 'top_speed_kmh', 'battery_capacity_kWh', alpha = 0.7)
# g.set_titles(col_template='{col_name}', row_template='{row_name}')
# g.set(xlim=(100, 350), ylim=(0, 200))


g = sns.FacetGrid(df, col = 'segment')
g.map(sns.histplot, 'acceleration_0_100_s', bins = 5)
g.set_titles(col_template='{col_name}', row_template='{row_name}')
g.set_xlabels(fontsize = 8)
plt.show()



