Causal inference to investigate french electricity market
==============================

structural causal model of nuclear availebility in France

Main notebooks are 01_get_data.ipynb for data acquisition, 
04_evaluate_scm.py for creation, fit and evaluation of structured causal model via DoWhy 
and 05_visualize_evaluation.ipynb for the visualization of the results from script 04. 
Notebooks 02 and 03 are more for visualization misc and exploration of the data. Notebook 06 explores the price difference between France and other countries.

Project Organization
------------

    |-- README.md                   <- The top-level README for developers using this project.
    |-- data
    |   |
    |   |-- raw                     <- The original, immutable data dump.
    |   |-- intermediate             <- files create using get_data.ipynb
    |   |-- processed               <- final data
    |
    |-- models                      <- Structural causal models. results from evaluate_scm notebook.
    |   |
    |   |
    |   |-- final_model             <- folder of specific model runthrough can be modified in scm_config.py
    |   |   |-- FR3                 <- results for the scm with price_da_FR as target
    |   |   |-- FR4                 <- results for the scm with net_export_FR  as target
    |   |   |-- ES3                 <- results for the scm with price_da_ES3 as target
    |   
    |-- notebooks                   <- Jupyter notebooks. Number for ordering. 
    |   |
    |   |-- scm_config.py           <- config file for model run, contains important parameters
    |   |-- scripts                 <- functions that the notebooks use. utils.py and countries.py are used by almost every 
    |                                   notebook.
    |
    |-- reports  
    |   |
    |   |-- figures                 <- Generated graphics and figures to be used in reporting. 
    |   |                               visualization of the causal model is in the folder "model_evaluation"  
    |
    |-- conda_env.yml                   <- The environment file for reproducing the analysis environment, e.g.
    │                                   generated with `pip freeze > requirements.txt` 
    |-- .env.txt                        <- Needs to be created, contains necessary API keys
    |-- data_selected_guide.xlsx            <- Table of all data sources  

Installation
------------

An environment can be created via conda_env.yml. We used miniforge.
The ENTSO-E transparency platform provides data needed for the analysis.
To get access to the ENTSO-E transperancy platform a .env.txt (saved at ../.env.txt) needs to be added containing an API-key (api_key_entsoe=xxxxx).
To register for ENTSO-E and get an API-key go to ([ENTSO-E transparency platform](https://www.entsoe.eu/data/transparency-platform/)).

First the folder structure needs to be constructed by running the second cell after imports "create necessary folders".
of notebook 01_get_data.ipynb.

Secondly some data needs to be downloaded manually.


- Installed nuclear capacity France
    - login into ENTSO-E transparency platform
    - download from  [ENTSO-E transparency platform](https://transparency.entsoe.eu/generation/installed/perType?permalink=691f25daabdb9a19ccd6eb73)
    - save as .csv file under path *../data/raw/ENTSO-E/installed_capacity_production_type_FR_2018_2020.csv*. (config.py paths["installed_cap_raw_FR_2018_2020"])
    - download from  [ENTSO-E transparency platform](https://transparency.entsoe.eu/generation/installed/perType?permalink=691f26583936c4051e5916e2)
    - save as .csv file under path *../data/raw/ENTSO-E/installed_capacity_production_type_FR_2021_2023.csv*. (config.py paths["installed_cap_raw_FR_2021_2023"])


- Filling rate hydro storage France
    - download from [ENTSO-E transparency platform](https://transparency.entsoe.eu/generation/hydroReserve?permalink=691f2828dc8b9d780681e46b)
    - save as .csv file under path *../data/raw/ENTSO_E/filling_rate_FR_2018_2020.csv* (config.py paths["filling_rate_raw_FR_2018_2020"])
    - download from [ENTSO-E transparency platform](https://transparency.entsoe.eu/generation/hydroReserve?permalink=691f28733936c4051e5916e6 )
    - save as .csv file under path *../data/raw/ENTSO_E/filling_rate_FR_2021_2023.csv* (config.py paths["filling_rate_raw_FR_2021_2023"])
- Filling rate hydro storage Spain
    - download from [ENTSO-E transparency platform](https://transparency.entsoe.eu/generation/hydroReserve?permalink=691f28b43936c4051e5916e9)
    - save as .csv file under path *../data/raw/ENTSO_E/filling_rate_ES_2018_2020.csv* (config.py paths["filling_rate_raw_ES_2018_2020"])
    - download from [ENTSO-E transparency platform](https://transparency.entsoe.eu/generation/hydroReserve?permalink=691f290cdc8b9d780681e5f6)
    - save as .csv file under path *../data/raw/ENTSO_E/filling_rate_ES_2021_2023.csv* (config.py paths["filling_rate_raw_ES_2021_2023"])
- River temp of Spain
    - Ebro river is used as proxy
    - dowwnload from [SAIH Ebro](https://www.saihebro.com/datos/historicos)
    - select Tag types: "Temperature del Agua"
    - select all stations and all tags and specify the period of 01.01.2018-01.01.2024 and daily statistics
    - save as .csv files (dot decimal) under path *../data/raw/DatosHistoricos_Ebro_temp* (config.py paths["river_temp_ES_raw"])
- River flow of Spain
    - Ebro river is used as proxy
    - downloaded from [SAIH Ebro](https://www.saihebro.com/datos/historicos)
    - select Tag types: "Causal aforo de rio"
    - select all stations and all tags of "Rio Ebro" and specify the period of 01.01.2018-01.01.2024 and daily statistics
    - save as .csv files (dot decimal) under path *../data/raw/DatosHistoricos_Ebro_flow* (config.py paths["river_flow_ES_raw"])


- ERA 5 data France
    - Grid to approximate France: latitudes 41◦N - 52◦N and longitudes 6◦W - 10◦E
    - download 100 m u and v component wind speed from ERA5 for France and save under *../data/raw/ERA5/wind_FR/data.grib* (config.py paths["era5_wind_raw_FR"])
    - download 2m temp and solar radiation downwards from ERA5 for France and save under *../data/raw/ERA5/solar_temp_FR/data.grib* (config.py paths["era5_solar_temp_raw_FR"])
- ERA 5 data Spain
    - Grid to approximate Spain:atitudes 36◦N - 44◦N and longitudes 10◦W - 5◦E
    - download 100 m u and v component wind speed from ERA5 for Spain and save under *../data/raw/ERA5/wind_ES/data.grib* (config.py paths["era5_wind_raw_ES"])
    - download 2m temp and solar radiation downwards from ERA5 for Spain and save under *../data/raw/ERA5/solar_temp_ES/data.grib* (config.py paths["era5_solar_temp_raw_ES"])
- Nuclear availability France
    - Download from [RTE Inside Information Platform](https://iip.cloud-rte-france.com/production-unavailability) by selecting Fuel Type
    - save as .csv under *../data/raw/nuclear_rte/ProductionUnavailability.csv* (config.py paths["na_gen_unavail_rte_raw_FR"])
- Nuclear availability Spain
    - Download from [ESIOS red electrica](https://www.esios.ree.es/en/analysis/474?vis=1&start_date=31-12-2017T00%3A00&end_date=01-01-2024T23%3A55&compare_start_date=30-12-2017T00%3A00&groupby=hour)
    - save as .csv under *../data/raw/nuclear_esios/nuclear_esios.csv* (config.py paths["na_gen_unavail_esios_raw_ES"])
- Carbon price France and Spain
    - download both ETS from [worldbank.org](https://carbonpricingdashboard.worldbank.org/compliance/price)
    - save as .csv file under path *../data/raw/carbonprice.csv* (config.py paths["carbon_price_raw"])
- Gas price TTF
    - please contact authors for data access
    - save as .xlsx file under path *"../data/raw/Price+History_20241019_2220.xlsx"* (config.py paths["gas_price_raw"])
- Gas price MIBGAS
    - download from [MIBGAS](https://www.mibgas.es/en/file-access)
    - save as .xlsx files under path *../data/raw/MIBGAS/* (config.py paths["gas_price_MIBGAS_raw"])
- Public holidays of France
    - download from [data.gouv.fr](https://www.data.gouv.fr/en/datasets/jours-feries-en-france/) (Toutes zones)
    - save as .csv file under path *../data/raw/france_holiday.csv* (config.py paths["FR_holiday_raw"])
- Public holidays of Spain
    - An excel file has been created with two columns "Day" and "Holiday", where the first contains the date of holidays and the second is a description of the holiday
    - source for this file is [office holidays](https://www.officeholidays.com/countries/spain), only national holidays are included
    - the file is saved under path *../data/raw/spain_holiday.xlsx* (config.py paths["ES_holiday_raw"])
    - the file either be created itself or ask the authors for access
- Production of electricity and derived heat by type of fuel in GWh
    - download from [nrg_bal_peh](https://ec.europa.eu/eurostat/databrowser/explore/all/envir?lang=en&subtheme=nrg.nrg_quant.nrg_quanta&display=list&sort=category)
    - save as .tsv file under path *../data/raw/nrg_bal_peh_el_prod_by_fuel.tsv* (config.py paths["eurostat_raw"])
    
After that the rest of 01_get_data_nuc.ipynb can be executed. Execution times may vary with internet connection.
For reference in our case a complete runthrough takes 3-4h.
Also french river temperature and flow data collection can experience issues with server connection. 
If this happens try running the corresponding cells again.

After that all necessary data is provided.

The script 04_evaluate_scm.py can be used for creation, fit and evaluation of structured causal model via DoWhy. 
Hereby the script scm_config.py defines important parameters. For example the model_name, that defines the foldernames where results are saved.
Here the option falsification_max_num_samples_run, allows for a more precise falsification of the model if 2000 is used, while the standard evaluation at 500 is faster but less precise. Due to random permutations the exact number of LMC violations can vary in falsification, but this should not impact  that the original graph shows fewer LMC violations than the permuted graphs. In general running script 04_evaluate_scm.py requires significant RAM and time.
After running 04_evaluate_scm.py, 05_visualize_evaluation.ipynb can be used for the visualization of the results.

Authors & Acknowledgements
-------------------
- Anton Tausendfreund - software developement
- Sarah Schreyer - software developement

#### Special Thanks

- to the DoWhy library 
    - Amit Sharma, Emre Kiciman. DoWhy: An End-to-End Library for Causal Inference. 2020. https://arxiv.org/abs/2011.04216

    - Patrick Blöbaum, Peter Götz, Kailash Budhathoki, Atalanti A. Mastakouri, Dominik Janzing. DoWhy-GCM: An extension of DoWhy for causal inference in graphical causal models. 2024. MLOSS 25(147):1−7. https://jmlr.org/papers/v25/22-1258.html

