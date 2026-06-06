import pandas as pd
import numpy as np
import pickle
import os
import sys
sys.path.append("./")
sys.path.append("../")

from utils.feature_configuration import edges_FR_price, edges_FR_export, edges_ES_price

skip_plot = False

file_path = './data/data_selected_2018-2023.csv'
data = pd.read_csv(file_path)
# add timestamp index
data.index = pd.to_datetime(data['timestamp'])

columns_FR = []
for edge in edges_FR_price:
    columns_FR.append(edge[0])
    columns_FR.append(edge[1])
columns_FR = np.unique(columns_FR).tolist()

for edge in edges_FR_export:
    columns_FR.append(edge[0])
    columns_FR.append(edge[1])
columns_FR = np.unique(columns_FR).tolist()

columns_ES = []
for edge in edges_ES_price:
    columns_ES.append(edge[0])
    columns_ES.append(edge[1])
columns_ES = np.unique(columns_ES).tolist()

len(columns_FR), len(columns_ES)
for col in data.columns:
    if (col not in columns_FR) and (col not in columns_ES):
        print('Column not used in FR and ES analysis:', col)


# Prepare data for price models and export model (for France only, as we don't use an export model for Spain)
prev_col = data.shape[0]
X_FR = data.loc[~data[columns_FR].isna().any(axis=1), columns_FR].copy()
print("Number of rows without (was 'with' before !?) NA in selected columns for FR price model: ", X_FR.shape[0], " out of ", prev_col)
print('Keep {}% of the data for FR price model.'.format(round(X_FR.shape[0]/prev_col*100,2)))
timestamp = X_FR.index
X_FR.loc[:,'day_of_year_sin'] = np.sin(timestamp.dayofyear/365*2*np.pi)
X_FR.loc[:,'day_of_year_cos'] = np.cos(timestamp.dayofyear/365*2*np.pi)
X_FR.loc[:,'hour_sin'] = np.sin(timestamp.hour/24*2*np.pi)
X_FR.loc[:,'hour_cos'] = np.cos(timestamp.hour/24*2*np.pi)
y_FR_price = X_FR.loc[:, 'price_da_FR']
y_FR_export = X_FR.loc[:, 'net_export_FR']
X_FR = X_FR.drop(columns=['price_da_FR', 'net_export_FR'])

X_ES = data.loc[~data[columns_ES].isna().any(axis=1), columns_ES].copy()
print("Number of rows without NA in selected columns for ES price model: ", X_ES.shape[0], " out of ", prev_col)
print('Keep {}% of the data for ES price model.'.format(round(X_ES.shape[0]/prev_col*100,2)))
timestamp = X_ES.index
X_ES.loc[:,'day_of_year_sin'] = np.sin(timestamp.dayofyear/365*2*np.pi)
X_ES.loc[:,'day_of_year_cos'] = np.cos(timestamp.dayofyear/365*2*np.pi)
X_ES.loc[:,'hour_sin'] = np.sin(timestamp.hour/24*2*np.pi)
X_ES.loc[:,'hour_cos'] = np.cos(timestamp.hour/24*2*np.pi)
y_ES_price = X_ES.loc[:, 'price_da_ES']
X_ES = X_ES.drop(columns=['price_da_ES'])

directory = './data'
if os.path.exists(directory):
    print('Directory {} already exists.'.format(directory))
if not os.path.exists(directory):
    os.makedirs(directory)

y_FR_price.to_csv('{}/y_FR_price_full.csv'.format(directory), sep=',', index=True)
y_ES_price.to_csv('{}/y_ES_price_full.csv'.format(directory), sep=',', index=True)
y_FR_export.to_csv('{}/y_FR_export_full.csv'.format(directory), sep=',', index=True)
X_FR.to_csv('{}/X_FR_full.csv'.format(directory), sep=',', index=True)
X_ES.to_csv('{}/X_ES_full.csv'.format(directory), sep=',', index=True)

# Save features that have a direct edge in the causal graph to the target variable, 
# to be used for a "target model" (the model predicting the target variable) in the Shapley flow analysis
features_target_model = {}
features_target_model['FR_price'] = []
for node1,node2 in edges_FR_price:
    if node2 == 'price_da_FR':
        features_target_model['FR_price'].append(node1)
features_target_model['FR_export'] = []
for node1,node2 in edges_FR_export:
    if node2 == 'net_export_FR':
        features_target_model['FR_export'].append(node1)
features_target_model['ES_price'] = []
for node1,node2 in edges_ES_price:
    if node2 == 'price_da_ES':
        features_target_model['ES_price'].append(node1)

directory = './data'
if not os.path.exists(directory):
    os.makedirs(directory)
    
with open('{}/features_target_model.pkl'.format(directory), 'wb') as f:
    pickle.dump(features_target_model, f)