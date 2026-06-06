import sys
sys.path.append('../../.')

from pathlib import Path
# Resolve parent of this file (file's directory → parent)
PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT))
sys.path.append('../')
sys.path.append('./')

import pandas as pd

import xgboost as xgb

from utils.flow_adapted import CausalLinks, build_feature_graph, GraphExplainer, edge_credits2edge_credit, translator, create_xgboost_f
from utils.flow_adapted import build_feature_graph
from utils.helper_functions import calculate_edge_credit, read_csv_incl_timeindex
from utils.feature_configuration import edges_FR_price, edges_FR_export, edges_ES_price, additional_nuc_avail


import time
import dill
import tqdm
import multiprocess as mp
import os

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate Shapley flow edge credits for GBT model')
    parser.add_argument('--target', type=str, default='FR_price', help='Target variables to explain. Options are: FR_price, FR_export, ES_price')    
    parser.add_argument('--what_if', action="store_true", help='Whether to calculate Shapley flow edge credits for What-if scenarios instead of actual Shapley flow edge credits')
    return parser.parse_args()

args = parse_args()
print(args)

target_names = ['price_da_FR', 'net_export_FR', 'price_da_ES']

periods = [('2018-01-01', '2023-12-31')]

targets = [args.target]


for target in targets:
    print(target)
    if target == 'FR_price':
        edges = edges_FR_price
    elif target == 'FR_export':
        edges = edges_FR_export
    elif target == 'ES_price':
        edges = edges_ES_price

    for start_date, end_date in periods:
        model_name = 'xgb_{}_start_{}_end_{}'.format(target, start_date, end_date)

        X_full = read_csv_incl_timeindex('./data/X_full_{}.csv'.format(model_name))
        X_train = read_csv_incl_timeindex('./data/X_train_{}.csv'.format(model_name))
        X_test = read_csv_incl_timeindex('./data/X_test_{}.csv'.format(model_name))

        X_test_features_target = read_csv_incl_timeindex('./data/X_test_features_target_{}.csv'.format(model_name))

        if args.what_if:    
            print('Calculate what-if Shapley flow edge credits for model: {}'.format(model_name))
            if target == 'FR_price':
                    additional_nuc_avail = 10000
                    X_test = X_test[(X_test.index >= pd.to_datetime('2022-01-01 00:00:00', utc=True)) & (X_test.index < pd.to_datetime('2023-01-01 00:00:00', utc=True))]
                    X_test['nuclear_avail_rte_FR'] += additional_nuc_avail
            elif target == 'ES_price':
                file_path = './data/data_selected_2018-2023.csv'
                dataset_all_features = pd.read_csv(file_path, index_col=0, parse_dates=True)
                X_test = X_test[(X_test.index >= pd.to_datetime('2022-06-15 00:00:00', utc=True)) & (X_test.index < pd.to_datetime('2023-02-27 00:00:00', utc=True))]
                X_test['gas_price_ES'] = dataset_all_features.loc[X_test.index, 'gas_price_MIBGAS']
            elif target == 'FR_export':
                print('No what-if scenario implemented for FR export, ' \
                'calculating actual Shapley flow edge credits instead.')

        model = xgb.Booster()
        model.load_model("./models/{}_best.json".format(model_name))
        seed = 7
        
        n_bg = 96 # number of sampled background samples
        nsamples = 1000 # number of forefround samples to explain
        nruns = 750 # number of runs for Shapley flow (number of permutations to sample for Shapley value estimation)

        # choose background samples from training set and foreground samples from test set (this is consistent with the Shapley flow framework)
        bg = X_train.sample(n=n_bg, random_state=seed) # background samples
        fg = X_test.sample(n=nsamples, random_state=seed) # foreground samples (samples to explain)

        # save foreground and background samples for later use (e.g. for plotting)
        if not args.what_if:
            bg.to_csv('./credit_flow/bg_{}.csv'.format(model_name), sep=',', index=True)
            fg.to_csv('./credit_flow/fg_{}.csv'.format(model_name), sep=',', index=True)
        else:
            bg.to_csv('./credit_flow/what_if_scenarios/bg_{}.csv'.format(model_name), sep=',', index=True)
            fg.to_csv('./credit_flow/what_if_scenarios/fg_{}.csv'.format(model_name), sep=',', index=True)

        causal_links = CausalLinks()
        categorical_feature_names = []

        display_translator = translator(X_full.columns, X_full, X_full)

        feature_names = list(X_test.columns)
        feature_names_target = list(X_test_features_target.columns)
        
        # build causal graph from edges
        for edge in edges:
            node_cause = edge[0]
            node_effect = edge[1]
            if node_effect not in target_names:
                if not node_cause in feature_names:
                    print('Node_cause (parent node) not in feature_names: {}'.format(node_cause))
                causal_links.add_causes_effects(node_cause, node_effect)
        
        # add edges from features that have a direct effect on the target to the target, 
        # with the function being the trained GBT model (this is the "target model" in the Shapley flow analysis)
        causal_links.add_causes_effects(feature_names_target, 
                                            target,
                                            create_xgboost_f(feature_names_target, model))
        print(feature_names, '\n',feature_names_target)

        # build surrogate models for all nodes in the causal graph and calculate R2 scores for the surrogate models (this is just for evaluation, not necessary for Shapley flow calculation)
        causal_graph, r2_scores = build_feature_graph(X_full,
                                        causal_links=causal_links, 
                                        categorical_feature_names=categorical_feature_names,
                                        display_translator=display_translator,
                                        target_name=target,# target_name=target_name,
                                        method='xgboost')
        if not args.what_if:
            with open('./credit_flow/causal_graph_{}.pkl'.format(model_name), 'wb') as file:
                dill.dump(causal_graph, file)
            with open('./credit_flow/r2_scores_{}.pkl'.format(target), 'wb') as file:
                dill.dump(r2_scores, file)
        else:
            with open('./credit_flow/what_if_scenarios/causal_graph_{}.pkl'.format(model_name), 'wb') as file:
                dill.dump(causal_graph, file)
            with open('./credit_flow/what_if_scenarios/r2_scores_{}.pkl'.format(target), 'wb') as file:
                dill.dump(r2_scores, file)

        # calculate multiple background result (same as in income.ipynb in the Shapley Flow repository) 
        # in parallel to speed up calculation (this is the most time consuming part)
        # change this to a suitable value, depending on machine (e.g. 6, 12; on cluster 20)
        num_processes = 96 

        start = time.time()

        model.set_param('n_jobs', -1)
        model.set_param('device', 'cpu')

        pool = mp.Pool(num_processes)
        _args = [(causal_graph, bg[i:i+1], fg, nruns) for i in range(len(bg))]
        edge_credits = pool.starmap(calculate_edge_credit, tqdm.tqdm(_args, total=len(_args)))
        pool.close()
        pool.join()

        end = time.time()
        print('Calculation time: ', end - start)
        
        # We need this step for being able to draw shapley flow (need to call shap_values for one bg sample redundandly)
        model.set_param('n_jobs', 96) 
        explainer = GraphExplainer(causal_graph, bg[0:1], nruns, silent=False)
        cf = explainer.shap_values(fg)
        # save credit flow to file
        cf.edge_credit = edge_credits2edge_credit(edge_credits, cf.graph)
        
        if not args.what_if:
            directory = './credit_flow'
        else:
            directory = './credit_flow/what_if_scenarios'

        if not os.path.exists(directory):
            os.makedirs(directory)
        with open('{}/flow_{}.pkl'.format(directory, model_name), 'wb') as file:
            dill.dump(cf, file)