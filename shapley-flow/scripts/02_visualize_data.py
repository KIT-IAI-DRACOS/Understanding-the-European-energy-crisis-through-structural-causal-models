import sys
sys.path.append("./")
sys.path.append("../")

import pandas as pd
import matplotlib.pyplot as plt
from utils.helper_functions import read_csv_incl_timeindex



directory = './data'

X_FR = read_csv_incl_timeindex('{}/X_FR_full.csv'.format(directory))
y_FR_price = read_csv_incl_timeindex('{}/y_FR_price_full.csv'.format(directory))
y_FR_export = read_csv_incl_timeindex('{}/y_FR_export_full.csv'.format(directory))
X_ES = read_csv_incl_timeindex('{}/X_ES_full.csv'.format(directory))
y_ES_price = read_csv_incl_timeindex('{}/y_ES_price_full.csv'.format(directory))

file_path = './data/data_selected_2018-2023.csv'
dataset_all_features = pd.read_csv(file_path)
# add timestamp index
dataset_all_features.index = pd.to_datetime(dataset_all_features['timestamp'])

fig,ax = plt.subplots(2,1, figsize=(26,10))
y_FR_price.plot(ax = ax[0])
y_ES_price.plot(ax = ax[1])
fig.tight_layout()
fig.show()

y_FR_export.plot(figsize=(26,10))
plt.show()