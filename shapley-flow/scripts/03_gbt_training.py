import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import os
import pickle
from scipy.stats import randint, uniform

import sys
from pathlib import Path
# Resolve parent of this file (file's directory → parent)
PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT))
sys.path.append("./")
sys.path.append("../")

from utils.helper_functions import read_csv_between, read_csv_incl_timeindex

directory = './models'
if os.path.exists(directory):
    print("Directory {} already exists. Models will be overwritten.".format(directory))
    
periods = [('2018-01-01', '2023-12-31')]
countries = ['FR', 'ES']
for country in countries:
    if country == 'FR':
        targets = ['FR_price', 'FR_export']
    elif country == 'ES':
        targets = ['ES_price']
    for target in targets:
        for start_date, end_date in periods:
            model_name = 'xgb_{}_start_{}_end_{}'.format(target, start_date, end_date)
            X = read_csv_between('./data/X_{}_full.csv'.format(country), start_date, end_date)
            X['isworkingday_{}'.format(country)] = X['isworkingday_{}'.format(country)]*1.0 # fixes problem with boolean data types (by making boolean type a float)
            y = read_csv_between('./data/y_{}_full.csv'.format(target), start_date, end_date)



            # split data into test and train set using a 1-day sliding window split to prevent memorization of target 
            block_size = '1D'
            masker = [pd.Series(g.index) for n, g in X.groupby(pd.Grouper(freq=block_size))]
            train_mask, test_mask = train_test_split(masker, test_size = 0.2, random_state=21)

            X_full = X.copy()
            y_full = y.copy()

            X_train = X.loc[pd.concat(train_mask)]
            y_train = y.loc[pd.concat(train_mask)]
            X_test = X.loc[pd.concat(test_mask)]
            y_test = y.loc[pd.concat(test_mask)]
            
            # save full, train and test feature data for the entire dataset
            X_full.to_csv('./data/X_full_{}.csv'.format(model_name), sep=',', index=True)
            X_train.to_csv('./data/X_train_{}.csv'.format(model_name), sep=',', index=True)
            X_test.to_csv('./data/X_test_{}.csv'.format(model_name), sep=',', index=True)

            # save train and test data in order to calculate shap values later
            y_train.to_csv('./data/y_train_{}.csv'.format(model_name), sep=',', index=True)

            with open('./data/features_target_model.pkl', 'rb') as f:
                features_target_model = pickle.load(f)

            # save train and test data with only the features that have a direct edge to the target variable in the causal graph, 
            # for the "target model" in the Shapley flow analysis
            X_full = X_full[features_target_model[target]]
            X_train = X_train[features_target_model[target]]
            X_test = X_test[features_target_model[target]]

            X_full.to_csv('./data/X_full_features_target_{}.csv'.format(model_name), sep=',', index=True)
            X_train.to_csv('./data/X_train_features_target_{}.csv'.format(model_name), sep=',', index=True)
            X_test.to_csv('./data/X_test_features_target_{}.csv'.format(model_name), sep=',', index=True)
            y_full.to_csv('./data/y_full_{}.csv'.format(model_name), sep=',', index=True) # y is saved in two folders at the moment
            y_train.to_csv('./data/y_train_{}.csv'.format(model_name), sep=',', index=True)
            y_test.to_csv('./data/y_test_{}.csv'.format(model_name), sep=',', index=True)
            print(X_full.shape, X_train.shape, X_test.shape)

            masker_train = [pd.Series(g.index) for n, g in X_train.groupby(pd.Grouper(freq=block_size)) if len(g) > 0]
            # masker_train = [pd.Series(g.index) for n, g in X_train.resample('1D') if len(g) > 0]
            train_train_mask, train_val_mask = train_test_split(masker_train, test_size = 0.2, random_state=21)

            
            X_train_train = X_train.loc[pd.concat(train_train_mask)]
            y_train_train = y_train.loc[pd.concat(train_train_mask)]
            X_train_val = X_train.loc[pd.concat(train_val_mask)]
            y_train_val = y_train.loc[pd.concat(train_val_mask)]

            # conventional CV on 1 day window
            param_dist = {
                'max_depth': randint(3, 12),
                'learning_rate': uniform(0.01, 0.3),
                'subsample': uniform(0.5, 1),
                'min_child_weight': randint(1, 31),
                'reg_lambda': uniform(0, 1),
                'reg_alpha': uniform(0, 1),
                'n_estimators': randint(100, 1200),
            }

            param_fit = {
                'eval_set': [(X_train_val, y_train_val)],
            }
            xgb_model = xgb.XGBRegressor(objective='reg:squarederror', 
                                        verbosity=1, 
                                        n_jobs=50, 
                                        base_score = y_train.iloc[:, 0].mean(),
                                        random_state=42,
                                        early_stopping_rounds=50
                                        )
            random_search = RandomizedSearchCV(
                                        estimator=xgb_model, 
                                        param_distributions=param_dist, 
                                        cv=5, 
                                        n_iter=50, # 50 iterations for random search
                                        n_jobs=-1, 
                                        refit='neg_root_mean_squared_error', 
                                        random_state=42,
                                        pre_dispatch='2*n_jobs', 
                                        verbose=3,
                                        scoring=['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2'])
            
            random_search.fit(X_train_train, y_train_train, **param_fit)

            best_parameters = random_search.best_params_

            best_model = xgb.XGBRegressor(**best_parameters,
                                          random_state=42,
                                          objective='reg:squarederror',
                                          n_jobs=50,
                                          base_score = y_train.iloc[:, 0].mean()
                                          )
            
            # fit model with best hyperparameters on the whole training set (train + validation)
            best_model.fit(X_train, y_train)

            print("Best set of hyperparameters: ", best_parameters)
            print("Best score: ", random_search.best_score_)

            # save best model and best hyperparameters
            directory = './models'
            if not os.path.exists(directory):
                os.makedirs(directory)
            best_model.save_model('{}/{}_best.json'.format(directory, model_name))
            with open('{}/{}_best_hyperparameters.pkl'.format(directory, model_name), 'wb') as f:
                pickle.dump(best_parameters, f)