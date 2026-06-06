import importlib
import os

from scripts.constants import ENERGY_CRISIS, EC_BY_YEARS
from scripts.load_causal_results import *


def load_model(model_folder, data):

    config_path = os.path.join(model_folder, "config.py")
    module_name = "config"

    spec = importlib.util.spec_from_file_location(module_name, config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    loaded_model = {}
    for graph in config.graphs:
        graph_dict = {}
        graph_dict["model_folder"] = model_folder
        print(f"Loading graph: {graph['name']}")

        nodes = graph["nodes"]
        graph_dict["nodes"] = nodes
        causal_graph = graph["graph"]

        name = graph["name"]

        graph_dict["edges"] = graph["edges"]

        target = graph["target"][0]
        graph_dict["target"] = target
        print(f"Target: {target}")

        graph_data = data.loc[:, nodes].dropna()
        years = [f"{graph_data.index[0].year}-{graph_data.index[-1].year}"]

        data_by_time = {
            "before_ec": graph_data[:ENERGY_CRISIS],
            "during_ec": graph_data[ENERGY_CRISIS:],
            "total": graph_data,
        }
        graph_dict["data_by_time"] = data_by_time

        fig_dir = f"../reports/figures/model_evaluation/{config.model_name}/{name}/"
        graph_dict["fig_dir"] = fig_dir

        os.makedirs(fig_dir, exist_ok=True)

        print("Loading simple r2 scores...")
        r2_scores = load_r2_scores(graph=graph, years=years, model_folder=model_folder)
        graph_dict["r2_scores"] = r2_scores

        if config.with_coefficients:
            print("Loading coefficients...")

            coefficients = load_coefficients(
                graph_name=name, target=target, years=years, model_folder=model_folder
            )
            graph_dict["coefficients"] = coefficients

        # Not used in the end as test train is handled manually to allow weekly splits
        if config.with_evaluation:
            print("Loading evaluation... ")
            # r2 coefficients

            r2_scores, invertible_binary, invertible_values = load_evaluation(
                graph=graph,
                years=years,
                model_folder=model_folder,
            )
            graph_dict["r2_scores_evaluation"] = r2_scores
            graph_dict["invertible_binary"] = invertible_binary
            graph_dict["invertible_values"] = invertible_values

        if config.with_falsification:
            print("Loading falsification...")
            falsification = load_falsification(
                graph_name=name,
                years=years,
                model_folder=model_folder,
            )
            graph_dict["falsification"] = falsification

        loaded_model[graph["name"]] = graph_dict
    return loaded_model
