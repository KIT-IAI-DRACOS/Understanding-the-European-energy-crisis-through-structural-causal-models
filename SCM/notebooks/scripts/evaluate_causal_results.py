import pandas as pd
import pickle as pkl
from dowhy.graph import is_root_node


def compare_coefficients(graph_name: str, var: str, years: list) -> pd.DataFrame:
    """
    Load the results of the 04-evaluate_scm and create a DataFrame of structural coefficients for different times as columns.

    Parameters:
    graph_name (str): The name of the causal graph.
    var (str): The variable for which to compare coefficients.
    years (list): A list of years to compare.

    Returns:
    pd.DataFrame: A DataFrame containing the structural coefficients for different times as columns.
    """
    dir = f"../models/{graph_name}/"

    frames = []
    for t in years:
        with open(
            dir + f"coefficients/{graph_name}_{t}_coefficients.pkl", "rb"
        ) as handle:
            coefficients = pkl.load(handle)
        frames.append(
            pd.DataFrame.from_dict(coefficients[var], orient="index", columns=[t])
        )
    return pd.concat(frames, axis=1)


def compare_r2_scores(graph: dict, years: list) -> pd.DataFrame:
    """
    Create a DataFrame of R2 scores for different times as columns.
    This function reads evaluation results from serialized files, extracts the R2 scores for non-root
    nodes, and organizes them into a DataFrame.

    Parameters:
    graph (dict): The causal graph dictionary.
    years (list): A list of years to compare.

    Returns:
    pd.DataFrame: A DataFrame containing the R2 scores for different times as columns.
    """
    causal_graph = graph["graph"]
    graph_name = graph["name"]
    dir = f"../models/{graph_name}/"

    frames = []
    for t in years:
        with open(dir + f"evaluation/{graph_name}_{t}_evaluation.pkl", "rb") as handle:
            evaluate_causal_model = pkl.load(handle)
        performances = evaluate_causal_model.mechanism_performances
        r2_scores = {
            node: performances[node].r2
            for node in causal_graph.nodes
            if not is_root_node(causal_graph, node)
        }
        frames.append(pd.DataFrame.from_dict(r2_scores, orient="index", columns=[t]))
    return pd.concat(frames, axis=1).sort_index()
