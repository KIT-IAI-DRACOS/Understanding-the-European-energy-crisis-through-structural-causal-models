# miscellaneous functions used by many notebooks

import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle as pkl

from scripts.countries import country_codes, COUNTRY_CODE_TO_COUNTRY


def normalize(data, mean, std):
    """
    Normalize data by subtracting the mean and dividing by the standard deviation.

    Parameters:
    data (pandas.DataFrame or numpy.ndarray): The data to normalize.
    mean (float): The mean value.
    std (float): The standard deviation value.

    Returns:
    pandas.DataFrame or numpy.ndarray: The normalized data.
    """
    return (data - mean) / std


def read_file(path, column_names):
    """
    Read a CSV file, rename columns, convert to timestamp, and set as index.
    This function is used for forecasted and actual load and for renewable data, but not for generation data.

    Parameters:
    path (str): The path to the CSV file.
    column_names (dict): A dictionary mapping old column names to new column names.

    Returns:
    pandas.DataFrame: The processed DataFrame.
    """
    df_tmp = pd.read_csv(path).rename(columns=column_names)
    df_tmp["timestamp"] = pd.to_datetime(df_tmp["timestamp"], utc=True)
    df_tmp = df_tmp.set_index("timestamp")
    return df_tmp


def save_file(var, dir, filename):
    """
    Save a variable to a pickle file and a text file.

    Parameters:
    var (any): The variable to save.
    dir (str): The directory to save the files.
    filename (str): The name of the files (without extension).

    Returns:
    None
    """
    try:
        os.mkdir(dir)
    except OSError:
        pass
    with open(os.path.join(dir, filename + ".pkl"), "wb") as f:
        pkl.dump(var, f)
    with open(os.path.join(dir, filename + ".txt"), "w") as f:
        print(var, file=f)


def drop_duplicates(lst):
    """
    Drop duplicates from a list while preserving order.

    Parameters:
    lst (list): The list from which to drop duplicates.

    Returns:
    list: The list with duplicates removed.
    """
    return list(dict.fromkeys(lst))


def scale_font_latex(scaling_factor: float) -> None:
    """
    Scale font sizes for LaTeX-style plots.

    Parameters:
    scaling_factor (float): The scaling factor for font sizes.

    Returns:
    None
    """
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "font.serif": ["Arial"],
            "mathtext.fontset": "custom",
            "mathtext.sf": "Arial",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "pdf.fonttype": 42,
        }
    )

    SMALL_SIZE = 15 * scaling_factor
    MEDIUM_SIZE = 20 * scaling_factor
    BIGGER_SIZE = 30 * scaling_factor

    plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


COUNTRY_CODES_ALT = drop_duplicates([cc.split("_")[0] for cc in country_codes])


def split_column(df):
    """
    Split the first column of a DataFrame into multiple columns.

    Parameters:
    df (pandas.DataFrame): The DataFrame with the column to split.

    Returns:
    pandas.DataFrame: The DataFrame with the split columns.
    """
    col_name = df.columns[0]
    col = df[col_name]
    new_names = {k: v.split("\\", 1)[0] for k, v in enumerate(col_name.split(","))}
    df = (
        pd.concat([col.str.split(",", expand=True), df], axis=1)
        .rename(columns=new_names)
        .drop([col_name], axis=1)
    )
    return df


def read_eurostat_tsv(path):
    """
    Read a Eurostat TSV file, split the first column, filter by country codes, and rename columns.

    Parameters:
    path (str): The path to the TSV file.

    Returns:
    pandas.DataFrame: The processed DataFrame.
    """
    df = pd.read_csv(path, sep="\t", comment="#", na_values=[":", ": ", ": z"])
    df = split_column(df)
    df = df.loc[df["geo"].isin(COUNTRY_CODES_ALT)]
    df = df.rename(columns={col: col.replace(" ", "") for col in df.columns})
    return df


def convert_to_cc_index(df):
    """
    Convert a DataFrame to have country codes as the index.

    Parameters:
    df (pandas.DataFrame): The DataFrame to convert.

    Returns:
    pandas.DataFrame: The DataFrame with country codes as the index.
    """
    new_index = sorted(drop_duplicates(country_codes + COUNTRY_CODES_ALT))
    df = df[~df.index.duplicated(keep="first")]
    df = df.reindex(new_index)
    for k, v in COUNTRY_CODE_TO_COUNTRY.items():
        df.loc[k] = df.loc[v]
    df = df.drop(
        drop_duplicates(list(COUNTRY_CODE_TO_COUNTRY.values()) + ["IT_CALA", "IT_ROSN"])
    )
    return df
