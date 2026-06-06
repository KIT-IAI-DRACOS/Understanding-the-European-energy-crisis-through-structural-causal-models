from flow_adapted import GraphExplainer, node_dict2str_dict 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from flow_adapted import FlowDefaultDict
from collections import defaultdict
from collections.abc import Iterable
from pygraphviz import AGraph  
from utils.feature_configuration import paper_rename_dict, target_names, calendar_features, meteorological_features, energy_features, paper_rename_dict_including_units

def calculate_edge_credit(causal_graph, bg_i, fg, nruns, silent=True):
    """
    calculate edge credit using Shapley Flow method for one background sample (bg_i) and foreground sample (fg)
    
    Parameters:
    causal_graph: the causal graph built from the data and model, should be in the format of a dictionary where keys are parent nodes and values are dictionaries of child nodes and their corresponding functions
    bg_i: the background sample to calculate edge credit for, should be in the format of a dictionary where keys are feature names and values are feature values
    fg: the foreground sample to calculate edge credit for, should be in the format of a dictionary where keys are feature names and values are feature values
    nruns: the number of runs to calculate edge credit, should be an integer
    silent: whether to print progress messages, should be a boolean

    Returns:
    A dictionary where keys are parent nodes and values are dictionaries of child nodes and their corresponding edge credits.
    """

    cf_c = GraphExplainer(causal_graph, bg_i, nruns=nruns, silent=silent).shap_values(fg)
    return node_dict2str_dict(cf_c.edge_credit)

def read_csv_incl_timeindex(filepath: str) -> pd.DataFrame:
    """
    read csv file and set timestamp as index, return a dataframe
    
    Parameters:    filepath: the path to the csv file, should be a string
    Returns: a pandas DataFrame with the timestamp as the index
    """

    df = pd.read_csv(filepath)
    df.index = pd.to_datetime(df['timestamp'])
    df.drop('timestamp', axis=1, inplace=True)
    return df

def read_csv_between(filepath: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = read_csv_incl_timeindex(filepath)
    return df[start_date:end_date]




def get_color(target: str) -> str:
    if target == None:
        color = '#0173b2'
    elif target == 'FR_price':
        color = '#0173b2'
    elif target == 'FR_export':
        color = '#de8f05'
    elif target == 'ES_price':
        color = '#029e73'
    else:
        color = '#0173b2'
    return color

target_features = target_names
def credit2dot_pygraphviz(edge_credit, format_str, idx=-1,
                            max_display=None, show_fg_val=True, rankdir = "TB", fold_noise = True, 
                            show_CI = False, color = 'blue', dict_correlation_for_color = None, 
                            show_only_some_feature_selection = False):
    '''
    Function modified from the original function in shapflow (https://github.com/nathanwang000/Shapley-Flow)

    pygraphviz version of credit2dot 

    idx: the index of target to visualize, if negative assumes sum
    '''     
    G = AGraph(directed=True, rankdir=rankdir)

    edge_values = []
    max_v = 1e-6 # avoid division by zero error
    for node1, d in edge_credit.items():
        for node2, val in d.items():

            if fold_noise and node1.is_noise_node:
                continue # only visualize non noise node
            
            max_v = max(abs(val), max_v)
            edge_values.append(abs(val))

    edge_values = sorted(edge_values)
    if max_display is None or max_display >= len(edge_values):
        min_v = 0
    else:
        min_v = edge_values[-max_display]
        
    for node1, d in edge_credit.items():
        for node2, val in d.items():
            
            v = val
            if type(edge_credit) == FlowDefaultDict and show_CI:
                '''
                following eqn 12 of https://link.springer.com/chapter/10.1007/978-3-030-57321-8_2
                '''
                vals = [ec[node1.name][node2.name] for ec in edge_credit.ecs]
                std = np.std(vals)
                n = len(edge_credit.ecs)
                a = 1.96 * std/np.sqrt(n)
                l, r = v - a, v + a
                edge_label = f"({l:.2f}, {v:.2f}, {r:.2f})"
            else:
                edge_label = format_str.format(v)
            if abs(v) < min_v: continue
            if fold_noise and node1.is_noise_node:
                continue # only visualize non noise nodes

            red = "#ff0051"
            blue = "#008bfb"
            black = "#000000"
            if idx < 0:
                color = color
            else:
                color = f"{blue}ff" if v < 0 else f"{red}ff" 
            
            max_w = 7
            min_w = 0.5
            width = abs(v) / max_v * (max_w - min_w) + min_w
            
            if node1.is_dummy_node:
                continue 

            if node2.is_dummy_node: # use the only direct child
                node2 = node2.children[0]
            if show_only_some_feature_selection:
                condition1 = (node1.name in ['air_temp_era5_FR',  'load_da_FR', 'nuclear_avail_esios_ES',
                                    'nuclear_avail_rte_FR', 'river_flow_mean_FR', 'river_temp_FR', 'run_off_gen_FR', 
                                    'air_temp_era5_ES', 'load_da_ES', 'nuclear_avail_esios_ES',
                                    'nuclear_avail_rte_FR', 'river_flow_mean_ES', 
                                    'river_temp_ES', 'run_off_gen_ES', ])
                condition2 = (node2.name in ['FR_price', 'FR_export', 'ES_price', 'load_da_FR', 'nuclear_avail_esios_ES', 
                                                                                            'load_da_ES','nuclear_avail_rte_FR', 
                                                                                            'river_flow_mean_FR', 'river_flow_mean_ES',
                                                                                            'river_temp_FR', 'river_temp_ES',
                                                                                           'run_off_gen_FR', 'run_off_gen_ES',])
                condition = condition1 & condition2
            else:
                condition = True
            if condition:
                for node in [node1, node2]:
                    if node not in G:
                        if node.is_noise_node and fold_noise:
                            G.add_node(node, shape="point")
                        else:
                            shape = 'box' if node.is_categorical else 'ellipse'
                            if idx < 0 or \
                                not isinstance(node.target, np.ndarray) or \
                                node.is_noise_node:
                                G.add_node(node, label=paper_rename_dict[node.name], shape=shape)
                            else:
                                if show_fg_val:
                                    display_translator =defaultdict(lambda: (lambda x: x) )
                                    txt = display_translator[node.name](node.target[idx])
                                    if isinstance(txt, str):
                                        fmt = "{}: {}"
                                    else:
                                        fmt = "{}: " + format_str
                                    G.add_node(node, label=\
                                                fmt.format(node, txt),
                                                shape=shape)
                                else:
                                    G.add_node(node, label=paper_rename_dict[node.name], shape=shape)

                G.add_edge(node1, node2)
                e = G.get_edge(node1, node2)                
                e.attr["weight"] = 1
                e.attr["penwidth"] = width

                e.attr["pos"] = (20,20)

                e.attr["color"] = color
                e.attr["label"] = edge_label
                e.attr["fontsize"] = 28 
                min_c, max_c = 140, 255 
                alpha = "{:0>2}".format(hex(
                    int(abs(v) / max_v * (max_c - min_c) + min_c)
                )[2:]) 
                if idx < 0:
                    if dict_correlation_for_color is not None:
                        if dict_correlation_for_color['{}-{}'.format(node1.name, node2.name)] < 0:
                            e.attr["fontcolor"] = f"{red}{alpha}"
                        else:
                            e.attr["fontcolor"] = f"{black}{alpha}"
                    else:
                        e.attr["fontcolor"] = f"{blue}{alpha}"

                else:
                    e.attr["fontcolor"] = f"{blue}{alpha}" if v < 0 else\
                        f"{red}{alpha}"

    G.graph_attr.update(size = "9,15!") 

    G.node_attr.update(fontsize=28)
    G.layout(prog="dot")


    same_rank_calendar = G.add_subgraph(calendar_features, name="calendar_features")    
    same_rank_calendar.graph_attr.update(rank="same")

    same_rank_meteorological = G.add_subgraph(meteorological_features, name="meteorological_features")
    same_rank_meteorological.graph_attr.update(rank="same")

    same_rank_energy = G.add_subgraph(energy_features, name="energy_features")
    same_rank_energy.graph_attr.update(rank="same")

    return G


def credit2dot(raw_edge_credit,
                   format_str="{:.2f}", idx=-1,
                   max_display=None, show_fg_val=True, show_CI=False, rankdir = "TB", fold_noise = True, 
                   color = 'blue', dict_correlation_for_color = None, show_only_some_feature_selection = False):
        '''
Function modified from the original function in shapflow (https://github.com/nathanwang000/Shapley-Flow)

        convert the graph to pydot graph for visualization
        e.g.:
        G = cf.credit2dot()
        viz_graph(G)

        raw_edge_credit: edge_credit with potentiall multi edges
        idx: the index of target to visualize, if negative assumes sum,
             if an iterable, assumes sum over the list of values
        max_display: max number of edges attribution to display
        '''
        if type(raw_edge_credit) == FlowDefaultDict and show_CI:
            edge_credit = FlowDefaultDict(lambda: FlowDefaultDict(int))
            edge_credit.ecs = [defaultdict(lambda: defaultdict(int)) for _ in \
                               range(len(raw_edge_credit.ecs))]
        else:
            edge_credit = defaultdict(lambda: defaultdict(int))

        if isinstance(idx, Iterable):
            if len(idx) == 1:
                idx = idx[0]
                new_idx = idx
            else:
                new_idx = -1 # set downstream task to aggregate
        else:
            new_idx = idx
        
        # simplify for dummy intermediate node for multi-graph
        list_edges = []
        for node1, d in raw_edge_credit.items():
            for node2, val in d.items():
                if idx < 0:
                    edge_credit[node1][node2] += np.mean(np.abs(val))
                    list_edges.append(edge_credit[node1][node2])
                    
                else:
                    edge_credit[node1][node2] += val[idx]

        '''if normalize the edge credit by the max edge credit value, the graph will be more interpretable, but the absolute value will be lost'''
        for node1, d in raw_edge_credit.items():
            for node2, val in d.items():
                edge_credit[node1][node2] /= np.max(np.array(list_edges))
        return credit2dot_pygraphviz(edge_credit, format_str,
                                          new_idx, max_display,
                                          show_fg_val, rankdir = rankdir, fold_noise = fold_noise, 
                                          show_CI = show_CI, color = color, dict_correlation_for_color = dict_correlation_for_color,
                                          show_only_some_feature_selection = show_only_some_feature_selection)
def viz_graph(G):
    '''
    Function modified from the original function in shapflow (https://github.com/nathanwang000/Shapley-Flow)

    only applicable in ipython notebook setting 
    convert G (pygraphviz) to graphviz format and display with 
    ipython display
    '''
    from graphviz import Source    
    display(Source(G.string()))

def draw(idx=-1, max_display=None, format_str="{:.2f}",
            edge_credit=None, show_fg_val=True, rankdir = "TB", fold_noise = True, show_CI = False, 
            color = 'blue', dict_correlation_for_color = None, show_only_some_feature_selection = False):
    '''

    Function modified from the original function in shapflow (https://github.com/nathanwang000/Shapley-Flow)
    
    assumes using ipython notebook
    idx: the index of target to visualize, if negative assumes sum,
            if an iterable, assumes sum over the list of values
    '''
    if edge_credit is None:
        AssertionError("must provide edge_credit")
    dot = credit2dot(edge_credit,
                            format_str=format_str,
                            idx=idx,
                            max_display=max_display,
                            show_fg_val=show_fg_val,
                            rankdir = rankdir,
                            fold_noise=fold_noise,
                            show_CI=show_CI,
                            color = color,
                            dict_correlation_for_color = dict_correlation_for_color,
                            show_only_some_feature_selection = show_only_some_feature_selection)
    viz_graph(dot)
    return dot

def plot_dependency(name1, name2, cf, fg_values, color=None, figsize=(8, 7), x_label='', 
                    fig = None, axes = None, color_label='', scale_color=1, scale_x=1, scale_y = 1, label = None):
    '''
    plot the dependency between feature value and shapley flow value for the edge from name1 to name2
    
    Parameters:
    name1: the name of the parent node
    name2: the name of the child node
    cf: the causal flow object
    fg_values: the feature values
    color: the color of the points
    figsize: the size of the figure
    x_label: the label for the x-axis
    fig: the figure object
    axes: the axes object
    color_label: the label for the colorbar
    scale_color: the scale for the color
    scale_x: the scale for the x-axis
    scale_y: the scale for the y-axis
    label: the label for the points
    
    Returns:
    None
    '''
    for node1, d in cf.edge_credit.items():
        for node2, val in d.items():
            try:
                if node1.name == name1 and node2.name == name2:
                    if axes == None:
                        fig,ax = plt.subplots(figsize=figsize)
                    else:
                        ax = axes
                        fig = fig
                    df = fg_values[name1].copy()
                    df_shap = pd.DataFrame(val.flatten(), index=df.index, columns=['shapley-flow'])
                    df = pd.concat([df, df_shap], axis=1)
                    if name2 in fg_values.columns:
                        sc = ax.scatter(x=df[name1]/scale_x, y=df['shapley-flow']/scale_y, s=5, c=fg_values[name2]/scale_color, cmap='viridis')
                        colorbar = plt.colorbar(sc, ax=ax)
                        if color_label == '':
                            colorbar.set_label(paper_rename_dict_including_units[name2])
                        else: 
                            colorbar.set_label(paper_rename_dict_including_units[color_label])
                    else:
                        if color is None:
                            color = get_color(name2)
                        sc = ax.scatter(x=df[name1]/scale_x, y=df['shapley-flow']/scale_y, s=5, color = color, label = label)
                    if x_label == '':
                        ax.set_xlabel(paper_rename_dict_including_units[name1])
                    else: 
                        ax.set_xlabel(x_label)
            except Exception as e:
                raise Exception("Feature not found in graph! Original error: {}".format(e))
    return             


def calculate_correlation(name1, name2, cf, fg_values):
    '''
    calculate correlation between feature value and shapley flow value for the edge from name1 to name2
    
    Parameters:
    name1: the name of the parent node
    name2: the name of the child node
    cf: the causal flow object
    fg_values: the feature values
    
    Returns:
    float: correlation value
    '''
    for node1, d in cf.edge_credit.items():
        for node2, val in d.items():
            if node1.name == name1 and node2.name == name2:

                name1_old = name1
                df = fg_values[name1_old].copy()
                df_shap = pd.DataFrame(val.flatten(), index=df.index, columns=['shapley-flow'])
                df = pd.concat([df, df_shap], axis=1)
                corr = df[name1_old].corr(df['shapley-flow'])
                return corr
    raise Exception("Feature not found in graph!")

def rel_symmetric_difference(A: np.ndarray, F: np.ndarray) -> float:
    '''
    Calculate the Relative Symmetric Difference between two arrays.
    
    Parameters:
    A (array-like): Actual values.
    F (array-like): Forecasted values.
    
    Returns:
    float: Relative Symmetric Difference value.
    '''
    return 2 * np.abs(F - A) / (np.abs(A) + np.abs(F))

def smape(A: np.ndarray, F: np.ndarray) -> float:
    '''
    Calculate the Symmetric Mean Absolute Percentage Error (SMAPE) between two arrays.
    
    Parameters:
    A (array-like): Actual values.
    F (array-like): Forecasted values.
    
    Returns:
    float: SMAPE value.
    '''
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
