import numpy as np
import random
import sys
import os
import shutil
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from scripts.constants import ENERGY_CRISIS
from scm_config import *
from scripts.config import paths, model_paths
from scripts.utils import normalize
from scripts.causal_functions import (
    create_eval_scm,
    split_by_week,
    train_test_evaluation,
)

# Get absolute path
root_path = os.path.abspath(os.path.dirname(__file__))


np.random.seed(42)
random.seed(42)


# matplotlib.use("Agg")  # Use a non-interactive backend

data = pd.read_csv(os.path.join(root_path, paths["data_selected"])).set_index(
    "timestamp"
)
data.index = pd.to_datetime(data.index, utc="utc")

# transform isworkingday to numpy
data["isworkingday_FR"] = data["isworkingday_FR"].astype(int)
data["isworkingday_ES"] = data["isworkingday_ES"].astype(int)


# normalize data
normalized_data = data.copy()
for col in normalized_data.columns:
    normalized_data[col] = normalize(
        data=data[col], mean=data[col].mean(), std=data[col].std()
    )

# Create necessary directories
model_path = os.path.join(model_paths, model_name)
os.makedirs(model_path, exist_ok=True)
print(model_path)


for graph in graphs:
    print(graph["name"])
    create_eval_scm(
        selected_graph=graph,
        df_data=normalized_data,
        df_data_original=data,
        model_path=model_path,
        with_coefficients=with_coefficients,
        with_evaluation=with_evaluation,
        with_falsification=with_falsification,
        falsification_max_num_samples_run=falsification_max_num_samples_run,
        normalize=normalize,
        load_model=load_model,
    )

# The used config file is safed with the project for proper documentation
scm_config_file = "notebooks/scm_config.py"
model_config_file = f"models/{model_name}/config.py"

shutil.copy2(scm_config_file, model_config_file)
