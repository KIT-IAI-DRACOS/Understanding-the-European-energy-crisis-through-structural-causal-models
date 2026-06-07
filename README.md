This project is licensed under the Mozilla Public License 2.0.
It includes code from the PyWhy project, as well as from the shapflow project, which are licensed under the MIT License.
See `causal_plots.py` in `SCM` as well as `flow_adapted.py` in `shapley-flow` for details.

The project is split into two parts `SCM` and `shapley-flow`. 
Both parts need different environments. 
Hereby the code in the `SCM` folder should be executed first, as the file `01_get_data.ipynb` is used to create the datafile `data_selected_2018-2023.csv`, which is also used by `shapley-flow`.

For more information about the `SCM` and `shapley-flow` code please refer to the corresponding `README.md`.