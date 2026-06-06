import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from scipy.stats import randint, uniform
from dowhy import gcm
from dowhy.graph import is_root_node, get_ordered_predecessors
from dowhy.gcm.falsify import falsify_graph
from dowhy.gcm import kernel_based
from functools import partial

import numpy as np
import pandas as pd
from dowhy.gcm.fitting_sampling import fit

from scripts.utils import save_file


def split_by_week(data, test_size=0.2, random_state=42):
    """
    Split a DataFrame into train and test sets based on week periods."""
    # Ensure datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")

    # Assign week period
    data["week"] = data.index.to_period("W")

    # Get unique weeks
    unique_weeks = data["week"].unique()

    # Split weeks into train/test sets
    train_weeks, test_weeks = train_test_split(
        unique_weeks, test_size=test_size, random_state=random_state
    )

    # Assign all rows from each week accordingly
    train_df = data[data["week"].isin(train_weeks)].drop(columns="week")
    test_df = data[data["week"].isin(test_weeks)].drop(columns="week")

    return train_df, test_df


def create_causal_model(graph, data):
    """
    Create a linear Structural Causal Model (SCM) with an additive noise model.

    Parameters:
    graph (networkx.DiGraph): The causal graph.
    data (pandas.DataFrame): The normed data to fit the model.

    Returns:
    gcm.InvertibleStructuralCausalModel: The fitted causal model.
    """
    causal_model = gcm.InvertibleStructuralCausalModel(graph)
    nodes = list(causal_model.graph.nodes)

    print("Nodes:", nodes)
    for i, node in enumerate(nodes):
        print(f"Fitting node '{node}'...")
        if is_root_node(causal_model.graph, node):
            causal_model.set_causal_mechanism(node, gcm.EmpiricalDistribution())
        else:
            sklearn_model = gcm.ml.SklearnRegressionModel(LinearRegression())
            # Wrap it in an Additive Noise Model
            model = gcm.AdditiveNoiseModel(sklearn_model)
            causal_model.set_causal_mechanism(node, model)

    # Fit the model to the data
    gcm.fit(causal_model, data)

    return causal_model


def train_test_evaluation(selected_graph, data, test_size=0.2):
    """
    Fits a SCM to data based in a train test split
    and calculate r2 scores based on the test set.

    Parameters:
    selected_graph (dict): The Graph dict.
    data (pandas.DataFrame): The normed data to fit the model.
    test_size (float, optional): Fraction of test set. Default is 0.2.
    """
    nodes = selected_graph["nodes"]
    graph = selected_graph["graph"]
    name = selected_graph["name"]
    # adapted from the evaluate_causal_model function in dowhy
    data = data.loc[:, nodes].dropna()
    train_df, test_df = split_by_week(data, test_size=test_size)
    causal_model = create_causal_model(graph, train_df)

    return causal_model, train_df, test_df


def get_evaluation_on_test(
    causal_model, train_df, test_df, data_original, normalize=False
):
    """
    Returns R2, mean absolut error and root mean squared error of the causal model.
    """
    # Evaluate on test set
    r2_scores = {}
    mae_scores = {}
    rmse_scores = {}
    for node in causal_model.graph.nodes:
        if is_root_node(causal_model.graph, node):
            continue
        # Get the causal mechanism for the node
        causal_mechanism = causal_model.causal_mechanism(node)
        categorical = gcm.util.general.is_categorical(train_df[node])
        parent_data_test = test_df[
            get_ordered_predecessors(causal_model.graph, node)
        ].to_numpy()
        conditional_expectations = (
            gcm.model_evaluation._estimate_conditional_expectations(
                causal_mechanism, parent_data_test, categorical, 0
            )
        )
        r2_scores[node] = float(r2_score(test_df[node], conditional_expectations))
        if normalize:
            mae_scores[node] = float(
                mean_absolute_error(test_df[node], conditional_expectations)
                * data_original[node].std()
            )
            rmse_scores[node] = float(
                root_mean_squared_error(test_df[node], conditional_expectations)
                * data_original[node].std()
            )
        else:
            mae_scores[node] = float(
                mean_absolute_error(test_df[node], conditional_expectations)
            )
            rmse_scores[node] = float(
                root_mean_squared_error(test_df[node], conditional_expectations)
            )

    return r2_scores, mae_scores, rmse_scores


# get structural coefficients
def get_linear_coefficients(causal_model, data_original, normalize=False):
    """
    Get the structural coefficients of the linear SCM.

    Parameters:
    causal_model (gcm.StructuralCausalModel): The fitted causal model.
    data_original (pandas.DataFrame): The non-normed data used to fit the model.
    normalize (bool, optional): If normalized data was used in the fitting. Defaut False.

    Returns:
    dict: A dictionary containing the structural coefficients.
    """
    coefficients = {}
    for node in causal_model.graph.nodes:
        if is_root_node(causal_model.graph, node):
            continue
        # coefficients
        coef_ = causal_model.causal_mechanism(node).prediction_model.sklearn_model.coef_
        ordered_parents = get_ordered_predecessors(causal_model.graph, node)
        intercept_ = causal_model.causal_mechanism(
            node
        ).prediction_model.sklearn_model.intercept_
        if normalize:
            intercept_ = (
                intercept_ * data_original[node].std() + data_original[node].mean()
            )
        for i in range(len(ordered_parents)):
            # denormalize coefficients
            if normalize:
                coef_[i] = float(
                    coef_[i]
                    * data_original[node].std()
                    / data_original[ordered_parents[i]].std()
                )
                intercept_ = (
                    intercept_ - coef_[i] * data_original[ordered_parents[i]].mean()
                )
        labeled_coef = {
            ordered_parents[i]: float(coef_[i]) for i in range(len(ordered_parents))
        }
        # intercept
        labeled_coef["intercept"] = float(intercept_)
        # save in dict
        coefficients[node] = labeled_coef
    return coefficients


# main function. name of causal graph and data set as input. return results of the different evaluation methods.
def create_eval_scm(
    selected_graph,
    df_data,
    df_data_original,
    model_path,
    with_coefficients=False,
    with_evaluation=False,
    with_falsification=False,
    falsification_max_num_samples_run=50,
    normalize=False,
    load_model=True,
):
    """
    Create and evaluate a Structural Causal Model (SCM).

    Parameters:
    selected_graph (dict): The selected causal graph containing nodes and edges.
    df_data (pandas.DataFrame): The normed data to fit the model.
    df_data_original (pandas.DataFrame): The non-normed data used to fit the model.
    model_path (path): Path where model outputs are saved.
    with_coefficients (bool, optional): Whether to calculate and save the structural coefficients. Default is False.
    with_evaluation (bool, optional): Whether to evaluate the causal model. Default is False.
    with_falsification (bool, optional): Whether to perform falsification tests on the causal model. Default is False.
    falsification_max_num_samples_run (int, optional): Influences resolution and running time of falsification. Default is 50.
    normalize (bool, optional): If data is normalized before fitting. Default is False.

    Returns:
    None
    """
    nodes = selected_graph["nodes"]
    causal_graph = selected_graph["graph"]
    name = selected_graph["name"]
    target = selected_graph["target"]

    df_data = df_data.loc[:, nodes].dropna()
    df_data_original = df_data_original.loc[:, nodes].dropna()
    years = f"{df_data.index[0].year}-{df_data.index[-1].year}"
    dir = os.path.join(model_path, f"{name}")
    print(dir)
    if normalize:
        data = df_data
    else:
        data = df_data_original
    try:
        os.makedirs(dir)
    except OSError as error:
        print(error)
    # print(selected_graph["edges"])

    causal_model_path = os.path.join(dir, f"{name}_{years}_model")
    if load_model:
        try:
            with open(causal_model_path, "rb") as f:
                causal_model = pkl.load(f)
            print("loaded causal model from file.")
        except:
            print("loading causal model from file failed.")
            load_model = False

    if not load_model:
        print("creating causal model")
        causal_model, train_df, test_df = train_test_evaluation(
            selected_graph, data, test_size=0.2
        )
        print("saving causal model")
        with open(causal_model_path, "wb") as f:
            pkl.dump(causal_model, f)
    train_df, test_df = split_by_week(data, test_size=0.2)
    r2_scores, mae_scores, rmse_scores = get_evaluation_on_test(
        causal_model,
        train_df,
        test_df,
        data_original=df_data_original,
        normalize=normalize,
    )

    dir_r2_scores = os.path.join(dir, "r2_scores")
    save_file(r2_scores, dir=dir_r2_scores, filename=f"{name}_{years}_r2_scores")

    dir_mae_scores = os.path.join(dir, "mae_scores")
    save_file(mae_scores, dir=dir_mae_scores, filename=f"{name}_{years}_mae_scores")

    dir_rmse_scores = os.path.join(dir, "rmse_scores")
    save_file(rmse_scores, dir=dir_rmse_scores, filename=f"{name}_{years}_rmse_scores")

    if with_coefficients:
        print("get coefficients")
        coefficients = get_linear_coefficients(
            causal_model=causal_model,
            data_original=df_data_original,
            normalize=normalize,
        )
        dir_new = os.path.join(dir, "coefficients")
        save_file(coefficients, dir=dir_new, filename=f"{name}_{years}_coefficients")

    # main overview evaluation.
    if with_evaluation:
        print("evaluate")
        evaluate_causal_model = gcm.evaluate_causal_model(
            causal_model,
            data,
            compare_mechanism_baselines=True,
            evaluate_causal_structure=False,  # prevents use of falsification as it is seperated
            # config=gcm.model_evaluation.EvaluateCausalModelConfig( # modifications if using falsification
            #     falsify_graph_significance_level=0.05,
            #     n_jobs=8,
            #     max_num_permutations_falsify=50,
            # ),
        )
        print(evaluate_causal_model)
        dir_new = os.path.join(dir, "evaluation")
        save_file(
            evaluate_causal_model, dir=dir_new, filename=f"{name}_{years}_evaluation"
        )

    # more precise falsification than that of the evaluation function. but takes longer
    if with_falsification:
        print("falsification")
        falsification_result = falsify_graph(
            causal_graph=causal_graph,
            data=data,
            plot_histogram=False,
            suggestions=False,
            n_permutations=50,  # increased number of permutations to improve tests
            allow_data_subset=False,
            significance_level=0.05,
            n_jobs=8,
            independence_test=partial(
                kernel_based,
                use_bootstrap=False,
                max_num_samples_run=falsification_max_num_samples_run,
            ),
            conditional_independence_test=partial(
                kernel_based,
                use_bootstrap=False,
                max_num_samples_run=falsification_max_num_samples_run,
            ),
        )
        dir_new = os.path.join(dir, "falsification")
        save_file(
            falsification_result, dir=dir_new, filename=f"{name}_{years}_falsification"
        )
