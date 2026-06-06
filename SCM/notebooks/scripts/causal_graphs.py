import networkx as nx

"""
This script defines several causal graph models using NetworkX directed graphs (DiGraph).
Each model represents a specific causal structure with nodes and edges, where:
- Nodes represent variables in the causal model.
- Edges represent causal relationships between variables.
The models are saved as dictionaries containing:
    - `name`: The name of the graph.
    - `nodes`: A list of nodes (variables) in the graph.
    - `edges`: A list of directed edges (causal relationships) in the graph.
    - `graph`: A NetworkX DiGraph object representing the causal graph.
    -`target`: The target variable of the graph.
Models:

Note:
- The script uses NetworkX to create and manipulate directed graphs.
"""

year = ["year"]
day_of_year = ["day_of_year_sin", "day_of_year_cos"]
hour_of_day = ["hour_sin", "hour_cos"]
calendrical = hour_of_day + day_of_year
gas_price_FR = ["gas_price_FR"]
gas_price_ES = ["gas_price_ES"]
gas_price = {"FR": gas_price_FR, "ES": gas_price_ES}
fuel_prices_FR = ["carbon_price_FR", "gas_price_FR"]
fuel_prices_ES = ["carbon_price_ES", "gas_price_ES"]
fuel_prices = {"FR": fuel_prices_FR, "ES": fuel_prices_ES}
rl_neighbours_FR = ["rl_BE", "rl_ES", "rl_DE_LU", "rl_IT_NORD"]
loads_FR = rl_neighbours_FR + ["load_da_FR"]
rl_neighbours_ES = ["rl_FR", "rl_PT"]
loads_ES = rl_neighbours_ES + ["load_da_ES"]
rl_neighbours = {"FR": rl_neighbours_FR, "ES": rl_neighbours_ES}
loads = {"FR": loads_FR, "ES": loads_ES}
river_flow_FR = ["river_flow_mean_FR"]
river_flow_ES = ["river_flow_mean_ES"]
river_flow = {"FR": river_flow_FR, "ES": river_flow_ES}
river_temp_FR = ["river_temp_FR"]
river_temp_ES = ["river_temp_ES"]
river_temp = {"FR": river_temp_FR, "ES": river_temp_ES}
river_data_FR = ["river_temp_FR"] + river_flow_FR
river_data_ES = ["river_temp_ES"] + river_flow_ES
river_data = {"FR": river_data_FR, "ES": river_data_ES}
nuclear_avail_FR = ["nuclear_avail_rte_FR"]
nuclear_avail_ES = ["nuclear_avail_esios_ES"]
nuclear_avail = {"FR": nuclear_avail_FR, "ES": nuclear_avail_ES}
run_off_gen_FR = ["run_off_gen_FR"]
run_off_gen_ES = ["run_off_gen_ES"]
run_off_gen = {"FR": run_off_gen_FR, "ES": run_off_gen_ES}
river_dependent_gen_FR = nuclear_avail_FR + run_off_gen_FR
river_dependent_gen_ES = nuclear_avail_ES + run_off_gen_ES
river_dependent_gen = {
    "FR": river_dependent_gen_FR,
    "ES": river_dependent_gen_ES,
}
solar_da_FR = ["solar_da_FR"]
solar_da_ES = ["solar_da_ES"]
solar_da = {"FR": solar_da_FR, "ES": solar_da_ES}
wind_da_FR = ["wind_da_FR"]
wind_da_ES = ["wind_da_ES"]
wind_da = {"FR": wind_da_FR, "ES": wind_da_ES}
renew_da = {
    "FR": solar_da_FR + wind_da_FR,
    "ES": solar_da_ES + wind_da_ES,
}
air_temp = {"FR": ["air_temp_era5_FR"], "ES": ["air_temp_era5_ES"]}
is_working_day = {
    "FR": ["isworkingday_FR"],
    "ES": ["isworkingday_ES"],
}
hydro_reservoir_gen = {
    "FR": ["hydro_reservoir_gen_FR"],
    "ES": ["hydro_reservoir_gen_ES"],
}
filling_rate = {
    "FR": ["filling_rate_FR"],
    "ES": ["filling_rate_ES"],
}
solar_cap_FR = ["solar_cap_FR"]
solar_cap_ES = ["solar_cap_ES"]
solar_cap = {"FR": solar_cap_FR, "ES": solar_cap_ES}
wind_cap_FR = ["wind_cap_FR"]
wind_cap_ES = ["wind_cap_ES"]
wind_cap = {"FR": wind_cap_FR, "ES": wind_cap_ES}
nuclear_cap_FR = ["nuclear_cap_FR"]
nuclear_cap_ES = ["nuclear_cap_ES"]
nuclear_cap = {"FR": nuclear_cap_FR, "ES": nuclear_cap_ES}
capacities_FR = solar_cap_FR + wind_cap_FR + nuclear_cap_FR
capacities_ES = solar_cap_ES + wind_cap_ES + nuclear_cap_ES
capacities = {"FR": capacities_FR, "ES": capacities_ES}


pv_cf_FR = ["ssrd_FR"]
pv_cf_ES = ["ssrd_ES"]
pv_cf = {"FR": pv_cf_FR, "ES": pv_cf_ES}
wind_cf_FR = ["wind_speed_100m_FR"]
wind_cf_ES = ["wind_speed_100m_ES"]
wind_cf = {"FR": wind_cf_FR, "ES": wind_cf_ES}
renew_cf = {
    "FR": pv_cf_FR + wind_cf_FR,
    "ES": pv_cf_ES + wind_cf_ES,
}

# Targets
price_da_FR = ["price_da_FR"]
price_da_ES = ["price_da_ES"]
net_export_FR = ["net_export_FR"]
net_export_ES = ["net_export_ES_MA"]


def create_graph(edges, graph_name, target):
    nodes = list(set(node for edge in edges for node in edge))
    graph = {
        "name": graph_name,
        "nodes": nodes,
        "edges": edges,
        "target": target,
        "graph": nx.DiGraph(edges),
    }
    return graph


def get_edges(country_code, target):
    year_edges = [
        (a, b)
        for a in year
        for b in fuel_prices[country_code]
        + loads[country_code]
        + renew_cf[country_code]
        + air_temp[country_code]
        + river_data[country_code]
        + renew_da[country_code]
        + nuclear_avail[country_code]
        + run_off_gen[country_code]
    ]

    caldr_edges = [
        (a, b)
        for a in calendrical
        for b in loads[country_code]
        + nuclear_avail[country_code]
        + renew_cf[country_code]
        + air_temp[country_code]
    ]
    day_edges = [
        (a, b)
        for a in day_of_year
        for b in river_data[country_code]
        + filling_rate[country_code]
        + gas_price[country_code]
    ]
    is_working_day_edges = [
        (a, b)
        for a in is_working_day[country_code]
        for b in loads[country_code] + nuclear_avail[country_code]
    ]
    temp_edges = [
        (a, b)
        for a in air_temp[country_code]
        for b in river_temp[country_code]
        + loads[country_code]
        + solar_da[country_code]
        + filling_rate[country_code]
    ]
    river_edges = [
        (a, b)
        for a in river_data[country_code]
        for b in river_dependent_gen[country_code] + filling_rate[country_code]
    ]
    cf_edges = (
        [(a, b) for a in pv_cf[country_code] for b in solar_da[country_code]]
        + [(a, b) for a in wind_cf[country_code] for b in wind_da[country_code]]
        + [(a, b) for a in renew_cf[country_code] for b in rl_neighbours[country_code]]
    )
    target_edges = [
        (a, b)
        for a in loads[country_code]
        + river_dependent_gen[country_code]
        + renew_da[country_code]
        + fuel_prices[country_code]
        + filling_rate[country_code]
        for b in target
    ]
    edges = (
        caldr_edges
        + is_working_day_edges
        + temp_edges
        + target_edges
        + cf_edges
        + river_edges
        + day_edges
        + year_edges
    )
    return edges


GRAPH_FR1 = create_graph(get_edges("FR", price_da_FR), "FR1", price_da_FR)
GRAPH_FR2 = create_graph(get_edges("FR", net_export_FR), "FR2", net_export_FR)

edges_FR3 = (
    get_edges("FR", price_da_FR)
    + [(a, b) for a in nuclear_avail_ES for b in price_da_FR]
    + [
        (a, b)
        for a in year + calendrical + is_working_day["FR"]
        for b in nuclear_avail_ES
    ]
)
GRAPH_FR3 = create_graph(edges_FR3, "FR3", price_da_FR)

edges_FR4 = (
    get_edges("FR", net_export_FR)
    + [(a, b) for a in nuclear_avail_ES for b in net_export_FR]
    + [
        (a, b)
        for a in year + calendrical + is_working_day["FR"]
        for b in nuclear_avail_ES
    ]
)
GRAPH_FR4 = create_graph(edges_FR4, "FR4", net_export_FR)

edges_FR5 = get_edges("FR", price_da_FR) + [
    (a, b) for a in nuclear_avail_ES + ["gas_price_ES"] for b in price_da_FR
]
GRAPH_FR5 = create_graph(edges_FR5, "FR5", price_da_FR)

edges_FR6 = get_edges("FR", net_export_FR) + [
    (a, b) for a in nuclear_avail_ES + ["gas_price_ES"] for b in net_export_FR
]
GRAPH_FR6 = create_graph(edges_FR6, "FR6", net_export_FR)

edges_FR7 = edges_FR5 + [(a, b) for a in calendrical for b in nuclear_avail_ES]
GRAPH_FR7 = create_graph(edges_FR7, "FR7", price_da_FR)
edges_FR8 = edges_FR6 + [(a, b) for a in calendrical for b in nuclear_avail_ES]
GRAPH_FR8 = create_graph(edges_FR8, "FR8", net_export_FR)

edges_FR9 = edges_FR3 + [(a, b) for a in hour_of_day for b in ["river_temp_FR"]]
GRAPH_FR9 = create_graph(edges_FR9, "FR9", price_da_FR)
edges_FR10 = edges_FR4 + [(a, b) for a in hour_of_day for b in ["river_temp_FR"]]
GRAPH_FR10 = create_graph(edges_FR10, "FR10", net_export_FR)

GRAPH_ES1 = create_graph(get_edges("ES", price_da_ES), "ES1", price_da_ES)
GRAPH_ES2 = create_graph(get_edges("ES", net_export_ES), "ES2", net_export_ES)

edges_ES3 = (
    get_edges("ES", price_da_ES)
    + [(a, b) for a in nuclear_avail_FR for b in price_da_ES]
    + [
        (a, b)
        for a in year + calendrical + is_working_day["ES"]
        for b in nuclear_avail_FR
    ]
)
GRAPH_ES3 = create_graph(edges_ES3, "ES3", price_da_ES)
edges_ES4 = (
    get_edges("ES", net_export_ES)
    + [(a, b) for a in nuclear_avail_FR for b in net_export_ES]
    # + [(a, b) for a in year for b in ["nuclear_avail_FR"]]
)
GRAPH_ES4 = create_graph(edges_ES4, "ES4", net_export_ES)


edges_ES5 = get_edges("ES", price_da_ES) + [
    (a, b) for a in nuclear_avail_FR + ["gas_price_FR"] for b in price_da_ES
]
GRAPH_ES5 = create_graph(edges_ES5, "ES5", price_da_ES)
edges_ES6 = get_edges("ES", net_export_ES) + [
    (a, b) for a in nuclear_avail_FR + ["gas_price_FR"] for b in net_export_ES
]
GRAPH_ES6 = create_graph(edges_ES6, "ES6", net_export_ES)

edges_ES7 = edges_ES5 + [
    (a, b) for a in ["run_off_gen_PT", "hydro_reservoir_gen_PT"] for b in price_da_ES
]
GRAPH_ES7 = create_graph(edges_ES7, "ES7", price_da_ES)
edges_ES8 = edges_ES6 + [
    (a, b) for a in ["run_off_gen_PT", "hydro_reservoir_gen_PT"] for b in net_export_ES
]
GRAPH_ES8 = create_graph(edges_ES8, "ES8", net_export_ES)

edges_ES9 = edges_ES5 + [
    (a, b) for a in ["run_off_gen_FR", "hydro_reservoir_gen_FR"] for b in price_da_ES
]
GRAPH_ES9 = create_graph(edges_ES9, "ES9", price_da_ES)
edges_ES10 = edges_ES6 + [
    (a, b) for a in ["run_off_gen_FR", "hydro_reservoir_gen_FR"] for b in net_export_ES
]
GRAPH_ES10 = create_graph(edges_ES10, "ES10", net_export_ES)

edges_ES11 = edges_ES5 + [
    (a, b)
    for a in [
        "run_off_gen_FR",
        "hydro_reservoir_gen_FR",
        "run_off_gen_PT",
        "hydro_reservoir_gen_PT",
    ]
    for b in price_da_ES
]
GRAPH_ES11 = create_graph(edges_ES11, "ES11", price_da_ES)
edges_ES12 = edges_ES6 + [
    (a, b)
    for a in [
        "run_off_gen_FR",
        "hydro_reservoir_gen_FR",
        "run_off_gen_PT",
        "hydro_reservoir_gen_PT",
    ]
    for b in net_export_ES
]
GRAPH_ES12 = create_graph(edges_ES12, "ES12", net_export_ES)
