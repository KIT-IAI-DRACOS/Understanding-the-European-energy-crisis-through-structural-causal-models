import pandas as pd

ENERGY_CRISIS = pd.Timestamp("20211001", tz="UTC")
START = pd.Timestamp("20180101T00", tz="utc")
END = pd.Timestamp("20231231T23", tz="utc")
QUERY_END = pd.Timestamp("20240103T00", tz="utc")
years = f"{START.year}-{END.year}"

TIME_DE_AT_LU_SPLIT = pd.Timestamp("20180930T21", tz="utc")

COUNTRY_CODE = "DE_LU"

TIME_IT_SPLIT_1 = pd.Timestamp("20181231T22", tz="utc")
TIME_IT_SPLIT_2 = pd.Timestamp("20201231T22", tz="utc")

TIME_CURRENCY_PLN = pd.Timestamp("20191119T22", tz="utc")
TIME_CURRENCY_RON = pd.Timestamp("20210617T21", tz="utc")

# Constant exchange rates
PLNToEUR = 0.23
RONToEUR = 0.21

# Spanish gas cap (EUR/MWh)
# Source: https://ec.europa.eu/commission/presscorner/detail/de/ip_23_2408
GAS_CAP_ES = {
    ("2022-06-01 00:00", " 2022-12-31 23:00"): 40,
    ("2023-01-01 00:00", "2023-01-31 23:00"): 45,
    ("2023-02-01 00:00", "2023-02-28 23:00"): 50,
    ("2023-03-01 00:00", "2023-03-31 23:00"): 55,
    ("2023-04-01 00:00", "2023-04-30 23:00"): 56.1,
    ("2023-05-01 00:00", "2023-05-31 23:00"): 57.2,
    ("2023-06-01 00:00", "2023-06-30 23:00"): 58.3,
    ("2023-07-01 00:00", "2023-07-31 23:00"): 59.4,
    ("2023-08-01 00:00", "2023-08-31 23:00"): 60.6,
    ("2023-09-01 00:00", "2023-09-30 23:00"): 61.7,
    ("2023-10-01 00:00", "2023-10-31 23:00"): 62.8,
    ("2023-11-01 00:00", "2023-11-30 23:00"): 63.9,
    ("2023-12-01 00:00", "2023-12-31 23:00"): 65,
}

INST_CAP_COLUMN_MAP = {
    "Biomass": "biomass_cap",
    "Energy storage": "storage_cap",
    "Fossil Brown coal/Lignite": "lignite_cap",
    "Fossil Coal-derived gas": "coal_derived_gas_cap",
    "Fossil Gas": "gas_cap",
    "Fossil Hard coal": "hard_coal_cap",
    "Fossil Oil": "oil_cap",
    "Fossil Oil shale": "oil_shale_cap",
    "Fossil Peat": "peat_cap",
    "Geothermal": "geothermal_cap",
    "Hydro Pumped Storage": "hydro_storage_cap",
    "Hydro Run-of-river and poundage": "run_off_cap",
    "Hydro Water Reservoir": "hydro_reservoir_cap",
    "Marine": "marine_cap",
    "Nuclear": "nuclear_cap",
    "Other": "other_cap",
    "Other renewable": "other_renew_cap",
    "Solar": "solar_cap",
    "Waste": "waste_cap",
    "Wind Offshore": "wind_off_cap",
    "Wind Onshore": "wind_on_cap",
    "Total Grand capacity": "total_cap",
}

EUROPEAN_BZN = [
    "AT",
    "BE",
    "CZ",
    "HR",
    "DK_1",
    "DK_2",
    "EE",
    "FI",
    "FR",
    "DE_LU",
    "DE_AT_LU",
    "HU",
    "IT_SICI",
    "IT_CSUD",
    "IT_CNOR",
    "IT_SUD",
    "IT_SARD",
    "IT_NORD",
    "IT_CALA",
    "IT_ROSN",
    "IT_BRNN",
    "IT_FOGN",
    "IT_PRGP",  # 4 hubs in Italy
    "LV",
    "LT",
    "NL",
    "NO_1",
    "NO_2",
    "NO_3",
    "NO_4",
    "NO_5",
    "PL",
    "PT",
    "RO",
    "SK",
    "SI",
    "ES",
    "SE_1",
    "SE_2",
    "SE_3",
    "SE_4",
]

GEN_COLUMN_MAP = {
    "Biomass Actual Aggregated": "biomass_gen",
    "Biomass Actual Consumption": "biomass_cons",
    "Fossil Brown coal/Lignite Actual Aggregated": "lignite_gen",
    "Fossil Brown coal/Lignite Actual Consumption": "lignite_cons",
    "Fossil Coal-derived gas Actual Aggregated": "coal_derived_gas_gen",
    "Fossil Coal-derived gas Actual Consumption": "coal_derived_gas_cons",
    "Fossil Gas Actual Aggregated": "gas_gen",
    "Fossil Gas Actual Consumption": "gas_cons",
    "Fossil Hard coal Actual Aggregated": "hard_coal_gen",
    "Fossil Hard coal Actual Consumption": "hard_coal_cons",
    "Fossil Oil Actual Aggregated": "oil_gen",
    "Fossil Oil Actual Consumption": "oil_cons",
    "Fossil Oil shale Actual Aggregated": "oil_shale_gen",
    "Fossil Peat Actual Aggregated": "peat_gen",
    "Geothermal Actual Aggregated": "geothermal_gen",
    "Geothermal Actual Consumption": "geothermal_cons",
    "Hydro Pumped Storage Actual Aggregated": "hydro_storage_gen",
    "Hydro Pumped Storage Actual Consumption": "hydro_storage_cons",
    "Hydro Run-of-river and poundage Actual Aggregated": "run_off_gen",
    "Hydro Run-of-river and poundage Actual Consumption": "run_off_cons",
    "Hydro Water Reservoir Actual Aggregated": "hydro_reservoir_gen",
    "Hydro Water Reservoir Actual Consumption": "hydro_reservoir_cons",
    "Marine Actual Aggregated": "marine_gen",
    "Nuclear Actual Aggregated": "nuclear_gen",
    "Nuclear Actual Consumption": "nuclear_cons",
    "Other Actual Aggregated": "other_gen",
    "Other Actual Consumption": "other_cons",
    "Other renewable Actual Aggregated": "other_renew_gen",
    "Other renewable Actual Consumption": "other_renew_cons",
    "Solar Actual Aggregated": "solar_gen",
    "Solar Actual Consumption": "solar_cons",
    "Waste Actual Aggregated": "waste_gen",
    "Waste Actual Consumption": "waste_cons",
    "Wind Offshore Actual Aggregated": "wind_off_gen",
    "Wind Offshore Actual Consumption": "wind_off_cons",
    "Wind Onshore Actual Aggregated": "wind_on_gen",
    "Wind Onshore Actual Consumption": "wind_on_cons",
    "B i o m a s s": "alt_biomass_gen",
}

# remove entries with consumption and remove "actual aggregated"
GEN_COLUMN_MAP_ALT = {
    k.replace(" Actual Aggregated", ""): v
    for k, v in GEN_COLUMN_MAP.items()
    if "Aggregated" in k
}


INST_CAP_COLUMN_MAP = {
    "Biomass": "biomass_cap",
    "Energy storage": "storage_cap",
    "Fossil Brown coal/Lignite": "lignite_cap",
    "Fossil Coal-derived gas": "coal_derived_gas_cap",
    "Fossil Gas": "gas_cap",
    "Fossil Hard coal": "hard_coal_cap",
    "Fossil Oil": "oil_cap",
    "Fossil Oil shale": "oil_shale_cap",
    "Fossil Peat": "peat_cap",
    "Geothermal": "geothermal_cap",
    "Hydro Pumped Storage": "hydro_storage_cap",
    "Hydro Run-of-river and poundage": "run_off_cap",
    "Hydro Water Reservoir": "hydro_reservoir_cap",
    "Marine": "marine_cap",
    "Nuclear": "nuclear_cap",
    "Other": "other_cap",
    "Other renewable": "other_renew_cap",
    "Solar": "solar_cap",
    "Waste": "waste_cap",
    "Wind Offshore": "wind_off_cap",
    "Wind Onshore": "wind_on_cap",
    "Total Grand capacity": "total_cap",
}

# better xlabels
VAR_NAMES_MAP = {
    "day_of_year_sin": "Day of year (sin)",
    "day_of_year_cos": "Day of year (cos)",
    "hour_sin": "Hour of day (sin)",
    "hour_cos": "Hour of day (cos)",
    "isworkingday_FR": "Is working day? (FR)",
    "isworkingday_ES": "Is working day? (ES)",
    "rl_BE": "RL BE",
    "rl_PT": "RL PT",
    "rl_FR": "RL FR",
    "rl_ES": "RL ES",
    "rl_IT_NORD": "RL IT-North",
    "rl_DE_LU": "RL DE-LU",
    "gas_price_FR": "Gas price FR",
    "load_da_FR": "Load day-ahead FR",
    "nuclear_avail_FR": "Nuclear availability FR",
    "nuclear_avail_rte_FR": "Nuclear availability FR",
    "run_off_gen_FR": "ROR generation FR",
    "price_da_FR": "Price day-ahead FR",
    "net_export_FR": "Net exports FR",
    "river_flow_mean_FR": "River flow rate FR",
    "river_temp_FR": "River temperature FR",
    "solar_da_FR": "Solar day-ahead FR",
    "air_temp_era5_FR": "Air temperature FR",
    "wind_da_FR": "Wind day-ahead FR",
    "carbon_price_FR": "Carbon price FR",
    "wind_speed_100m_FR": "100 m Wind speed FR",
    "ssrd_FR": "Surface solar radiation downwards FR",
    "filling_rate_FR": "Filling rate FR",
    "gas_price_ES": "Gas price ES",
    "load_da_ES": "Load day-ahead ES",
    "nuclear_avail_ES": "Nuclear availability ES",
    "nuclear_avail_esios_ES": "Nuclear availability ES",
    "run_off_gen_ES": "ROR generation ES",
    "price_da_ES": "Price day-ahead ES",
    "net_export_ES": "Net exports ES",
    "river_flow_mean_ES": "River flow rate ES",
    "river_temp_ES": "River temperature ES",
    "solar_da_ES": "Solar day-ahead ES",
    "air_temp_era5_ES": "Air temperature ES",
    "wind_da_ES": "Wind day-ahead ES",
    "carbon_price_ES": "Carbon price ES",
    "wind_speed_100m_ES": "100 m Wind speed ES",
    "ssrd_ES": "Surface solar radiation downwards ES",
    "filling_rate_ES": "Filling rate ES",
}

EC_BY_YEARS = {"2018-2021": "before_ec", "2021-2023": "during_ec", "2018-2023": "total"}
YEARS_BY_EC = {v: k for k, v in EC_BY_YEARS.items()}
EC_LEGEND = {
    "before_ec": "Before energy crisis",
    "during_ec": "During energy crisis",
    "total": "Total time",
}
