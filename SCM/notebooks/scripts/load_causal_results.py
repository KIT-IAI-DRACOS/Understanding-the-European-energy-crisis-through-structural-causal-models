import pandas as pd
import pickle as pkl
import os
import re
import logging
from dowhy.graph import is_root_node

# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def load_pickle(file_path: str):
    """
    Load a pickle file if it exists, otherwise log a warning.

    Parameters:
    file_path (str): The path to the pickle file.

    Returns:
    object: The loaded pickle object, or None if the file does not exist.
    """
    if os.path.exists(file_path):
        with open(file_path, "rb") as handle:
            return pkl.load(handle)
    else:
        logging.warning(f"File {file_path} does not exist.")
        return None


def load_coefficients(
    graph_name: str, target: str, years: list, model_folder: str
) -> pd.DataFrame:
    """
    Raises:
        KeyError: If the specified variable is not found in the coefficients for a given year.

    Load the results of the 04-evaluate_scm and create a DataFrame of structural coefficients for different times as columns.

    Parameters:
    graph_name (str): The name of the causal graph.
    target (str): The variable for which to compare coefficients.
    years (list): A list of years to compare.
    model_folder (str): Folder where model results are saved.

    Returns:
    pd.DataFrame: A DataFrame containing the structural coefficients for different times as columns.
    """
    dir = os.path.join(model_folder, graph_name)
    frames = []

    for t in years:
        file_path = os.path.join(dir, f"coefficients/{graph_name}_{t}_coefficients.pkl")
        coefficients = load_pickle(file_path)
        if coefficients is not None:
            if target in coefficients:
                frames.append(
                    pd.DataFrame.from_dict(
                        coefficients[target], orient="index", columns=[t]
                    )
                )
            else:
                raise KeyError(
                    f"Variable '{target}' not found in coefficients for year {t}."
                )
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def load_falsification(graph_name: str, years: list, model_folder: str) -> pd.DataFrame:
    """
    Load the results of the 04-evaluate_scm and create a DataFrame of falsification tests for different times as columns.

    Parameters:
    graph_name (str): The name of the causal graph.
    years (list): A list of years to compare.
    model_folder (str): Folder where model results are saved.

    Returns:
    pd.DataFrame: A DataFrame containing the falsification tests for different times as columns.
    """
    dir = os.path.join(model_folder, graph_name)
    frames = {}

    for t in years:
        file_path = os.path.join(
            dir, f"falsification/{graph_name}_{t}_falsification.pkl"
        )
        with open(file_path, "rb") as handle:
            falsification = pkl.load(handle)
            frames[t] = falsification
    return frames if frames else pd.DataFrame()


def load_r2_scores(graph: dict, years, model_folder):
    """
    Create a DataFrame of R2 scores for different times as columns.

    Parameters:
    graph (dict): The causal graph dictionary.
    years (list): A list of years to compare.
    model_folder (str): Folder where model results are saved.

    Returns:
    pd.DataFrame: A DataFrame containing the R2 scores for different times as columns.
    """
    causal_graph = graph["graph"]
    graph_name = graph["name"]
    dir = os.path.join(model_folder, graph_name)

    frames = []
    for t in years:
        file_path = os.path.join(dir, f"r2_scores/{graph_name}_{t}_r2_scores.pkl")
        r2_scores = load_pickle(file_path)
        r2_scores_filtered = {
            node: r2_score
            for node, r2_score in r2_scores.items()
            if not is_root_node(causal_graph, node)
        }
        frames.append(
            pd.DataFrame.from_dict(r2_scores_filtered, orient="index", columns=[t])
        )
    return pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()


def load_evaluation(graph: dict, years: list, model_folder: str) -> pd.DataFrame:
    # Not used in the end as test train is handled manually to allow weekly splits
    """
    Create a DataFrame of R2 scores for different times as columns.

    Parameters:
    graph (dict): The causal graph dictionary.
    years (list): A list of years to compare.
    model_folder (str): Folder where model results are saved.

    Returns:
    pd.DataFrame: A DataFrame containing the R2 scores for different times as columns.
    """
    causal_graph = graph["graph"]
    graph_name = graph["name"]

    dir = os.path.join(model_folder, graph_name)

    frames_r2 = []
    frames_binary = []
    frames_values = []
    for t in years:
        file_path = os.path.join(dir, f"evaluation/{graph_name}_{t}_evaluation.pkl")
        evaluate_causal_model = load_pickle(file_path)
        if evaluate_causal_model is not None:
            performances = evaluate_causal_model.mechanism_performances
            r2_scores = {
                node: performances[node].r2
                for node in causal_graph.nodes
                if not is_root_node(causal_graph, node)
            }
            frames_r2.append(
                pd.DataFrame.from_dict(r2_scores, orient="index", columns=[t])
            )
            results = evaluate_causal_model.pnl_assumptions
            frames_binary.append(
                pd.DataFrame.from_dict(
                    {node: results[node][1] for node in results.keys()},
                    orient="index",
                    columns=[t],
                )
            )
            frames_values.append(
                pd.DataFrame.from_dict(
                    {node: results[node][0] for node in results.keys()},
                    orient="index",
                    columns=[t],
                )
            )

    return (
        pd.concat(frames_r2, axis=1).sort_index() if frames_r2 else pd.DataFrame(),
        pd.concat(frames_binary, axis=1).sort_index(),
        pd.concat(frames_values, axis=1).sort_index(),
    )
