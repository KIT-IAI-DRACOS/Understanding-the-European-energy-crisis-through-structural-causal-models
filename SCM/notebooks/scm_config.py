# Config file for simulation run through

# Imports
import pandas as pd
from scripts.causal_graphs import *


model_name = "final_model"
graphs = [GRAPH_FR3, GRAPH_FR4, GRAPH_ES3]
load_model = False
normalize = True

with_coefficients = True
with_evaluation = True
with_falsification = True
falsification_max_num_samples_run = (
    2000  # should be set to a minimum of 500 for proper falsification 2000 is best
)
