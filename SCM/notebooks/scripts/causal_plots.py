import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os
import numpy as np
import seaborn as sns

from scripts.utils import scale_font_latex
from scripts.constants import VAR_NAMES_MAP, EC_BY_YEARS, YEARS_BY_EC, EC_LEGEND


def plot_coefficients_grouped(df, ax):
    heights = {
        "Before energy crisis": df["before_ec"],
        "During energy crisis": df["during_ec"],
        "Total time": df["total"],
    }

    x = np.arange(len(df.index))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0
    color = sns.color_palette("colorblind")

    for time, values in heights.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, values, width, label=time, color=color[multiplier])
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_xticks(x + width, df.index)
    ax.legend(
        loc="lower left",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        mode="expand",
        borderaxespad=0.0,
    )
    ax.grid(True)
    _ = plt.setp(
        ax.xaxis.get_majorticklabels(), rotation=45, ha="right", rotation_mode="anchor"
    )


def plot_coefficients(df, col, ax, color):
    """
    Plot a bar chart of structural coefficients.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the coefficients.
    col (str): The column name in the DataFrame to plot.
    ax (matplotlib.axes.Axes): The axes on which to plot the bar chart.
    color (str): The color of the bars.

    Returns:
    None
    """
    ax.bar(
        x=df.index,
        height=df[col],
        color=color,
    )
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    _ = plt.setp(
        ax.xaxis.get_majorticklabels(), rotation=45, ha="right", rotation_mode="anchor"
    )


def plot_evaluation_results_custom(
    evaluation_result, ax, bins=None, title="", savepath="", display=True
):
    """
    Plot a custom evaluation histogram.
    adapted from: from dowhy.gcm.falsify import plot_evaluation_results
    https://github.com/py-why/dowhy/blob/main/dowhy/gcm/falsify.py

    Parameters:
    evaluation_result (dowhy.gcm.falsify.FalsificationResult): The evaluation result to plot.
    ax (matplotlib.axes.Axes): The axes on which to plot the histogram.
    bins (int or sequence, optional): The bins for the histogram.
    title (str, optional): The title of the plot.
    savepath (str, optional): The path to save the plot.
    display (bool, optional): Whether to display the plot.

    Returns:
    tuple: A tuple containing the labels and p-values.
    """
    from dowhy.gcm.falsify import FalsifyConst, FALSIFY_METHODS

    COLORS = list(mcolors.TABLEAU_COLORS.values())

    # Plot histograms
    p_values = ""
    data = []
    labels = []

    evaluation_summary = {
        k: v for k, v in evaluation_result.summary.items() if k != FalsifyConst.MEC
    }
    for i, (m, m_summary) in enumerate(evaluation_summary.items()):
        data.append(m_summary[FalsifyConst.F_PERM_VIOLATIONS])
        labels.append(f"Violations of {FALSIFY_METHODS[m]} of permuted DAGs")
        p_values += (
            f"p-value {FALSIFY_METHODS[m]} = {m_summary[FalsifyConst.P_VALUE]:.2f}\n"
        )

    ax.hist(
        data,
        color=COLORS[: len(evaluation_summary)],
        bins=bins,
        alpha=0.5,
        label="",
        edgecolor="k",
    )

    # Plot given violations
    for i, (m, m_summary) in enumerate(evaluation_summary.items()):
        ylim = ax.get_ylim()[1]
        ax.plot(
            [m_summary[FalsifyConst.F_GIVEN_VIOLATIONS]] * 2,
            [0, ylim],
            "--",
            c=COLORS[i],
            label=f"Violations of {FALSIFY_METHODS[m]} of given DAG",
        )
        ax.set_ylim([0, ylim])

    ax.margins(0.05, 0)
    return labels, p_values


def plot_falsification_hist(falsification, fig_dir):
    # Falsification Histogram

    COLORS = list(mcolors.TABLEAU_COLORS.values())

    legend_elements = [
        Line2D(
            [0],
            [0],
            linestyle="--",
            color=COLORS[0],
            lw=1,
            label="Violations of TPa of given DAGs",
        ),
        Line2D(
            [0],
            [0],
            linestyle="--",
            color=COLORS[1],
            label="Violations of LMC of given DAGs",
        ),
        Patch(
            facecolor=COLORS[0],
            alpha=0.5,
            edgecolor="black",
            label="Violations of TPa of permuted DAGs",
        ),
        Patch(
            facecolor=COLORS[1],
            alpha=0.5,
            edgecolor="black",
            label="Violations of LMC of permuted DAGs",
        ),
    ]

    scale_font_latex(1.5)

    os.makedirs(fig_dir + "falsification/", exist_ok=True)
    p_values = []

    for t, evaluation_result in falsification.items():
        fig, ax = plt.subplots(1, 1, figsize=(16, 9), sharex=True)
        lable, p_value = plot_evaluation_results_custom(
            evaluation_result=evaluation_result, ax=ax
        )
        p_values.append(p_value)
        ax.set_xlabel("Fraction of violations")
        ax.set_ylabel("# Permutations")

        fig.legend(
            loc="upper left",
            frameon=False,
            bbox_to_anchor=(0.0, 1.3),
            borderaxespad=0.0,
            handles=legend_elements,
        )

        fig.savefig(
            fig_dir + f"falsification/falsification_histogram_custom_{t}.pdf",
            bbox_inches="tight",
        )
        plt.show()
        fig.clf()

    if len(falsification.keys()) == 3:
        fig, axes = plt.subplots(3, 1, figsize=(16, 16), sharex=True)
        fig.legend(
            loc="upper left",
            frameon=False,
            bbox_to_anchor=(0.0, 1.3),
            borderaxespad=0.0,
            handles=legend_elements,
        )
        for ax, key in zip(axes, falsification.keys()):
            lable, p_value = plot_evaluation_results_custom(
                evaluation_result=falsification[key], ax=ax
            )
            p_values.append(p_value)
            ax.set_xlabel("Fraction of violations")
            ax.set_ylabel("# Permutations")
            ax.annotate(
                text=EC_BY_YEARS[key], xy=(0.06, 0.79), xycoords="axes fraction"
            )

        fig.savefig(
            fig_dir + f"falsification/falsification_histogram_custom_grouped.pdf",
            bbox_inches="tight",
        )
        plt.show()
        fig.clf()


def plot_r2_scores(r2_scores, color, fig_dir):
    # r2-scores:
    scale_font_latex(1.25)
    os.makedirs(fig_dir + "falsification/", exist_ok=True)
    df = r2_scores
    df = df.rename(index=VAR_NAMES_MAP)
    for col in df.columns:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
        plot_coefficients(df=df, col=col, ax=ax, color=color)
        ax.set_ylabel("R2 score")
        ax.tick_params(axis="x", labelsize=16)
        fig.savefig(fig_dir + f"falsification/r2_scores_{col}.pdf", bbox_inches="tight")
        plt.show()
        ax.cla()
        fig.clf()
    if len(df.columns) == 3:
        # plot grouped coefficients
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
        df = df.rename(columns=EC_BY_YEARS)
        plot_coefficients_grouped(df=df, ax=ax)
        ax.set_ylabel("R2 score")
        ax.tick_params(axis="x", labelsize=16)
        fig.savefig(
            fig_dir + f"falsification/r2_scores_grouped.pdf", bbox_inches="tight"
        )
        plt.show()
        ax.cla()
        fig.clf()


def plot_total_coefficients(
    coefficients,
    data_by_time,
    unit,
    color,
    fig_dir,
    exports=False,
):
    os.makedirs(fig_dir + "coefficients/", exist_ok=True)
    scale_font_latex(1.5)
    # plot structure coefficient times delta_x

    df = coefficients.drop("intercept")
    df = df.rename(index=VAR_NAMES_MAP)
    delta_x = (
        (data_by_time["during_ec"].mean() - data_by_time["before_ec"].mean())
        .rename(index=VAR_NAMES_MAP)
        .loc[df.index]
    )
    df_delta_x = df.copy()
    for col in df.columns:
        df_delta_x[col] = df[col] * delta_x
        if exports:
            df_delta_x[col] = df_delta_x[col] / 1000
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
        plot_coefficients(df=df_delta_x, col=col, ax=ax, color=color)
        print(f"Plotting coefficients multiplied by delta_x:")
        print(df_delta_x)
        ax.tick_params(axis="x", labelsize=16)
        ax.set_ylabel(r"$c_{X} \cdot \Delta \overline{X} \;$ (" + unit + ")")
        fig.savefig(
            fig_dir + f"coefficients/coefficients_delta_{col}.pdf", bbox_inches="tight"
        )
        plt.show()
        ax.cla()
    fig.clf()

    if len(df.columns) == 3:
        # plot grouped coefficients
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
        df_delta_x = df.mul(delta_x, axis=0)
        if exports:
            df_delta_x = df_delta_x / 1000
        df_delta_x = df_delta_x.rename(columns=EC_BY_YEARS)
        plot_coefficients_grouped(df=df_delta_x, ax=ax)
        ax.tick_params(axis="x", labelsize=16)
        ax.set_ylabel(r"$c_{X} \cdot \Delta \overline{X} \;$ (" + unit + ")")
        fig.savefig(
            fig_dir + f"coefficients/coefficients_delta_grouped.pdf",
            bbox_inches="tight",
        )
        plt.show()
        ax.cla()

    # plot structure coefficient times std
    df = coefficients.drop("intercept")
    df = df.rename(index=VAR_NAMES_MAP)
    df_sigma = df.copy()
    for col in df.columns:
        sigma_x = (
            data_by_time[EC_BY_YEARS[col]]
            .std()
            .rename(index=VAR_NAMES_MAP)
            .loc[df.index]
        )
        df_sigma[col] = df[col] * sigma_x
        if exports:
            df_sigma[col] = df_sigma[col] / 1000
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
        plot_coefficients(df=df_sigma, col=col, ax=ax, color=color)
        print(f"Plotting coefficients multiplied by std for {col}:")
        print(df_sigma)
        ax.tick_params(axis="x", labelsize=16)
        ax.set_ylabel(r"$c_{X} \cdot \sigma_{X} \;$ (" + unit + ")")

        fig.savefig(
            fig_dir + f"coefficients/coefficients_std_{col}.pdf", bbox_inches="tight"
        )
        plt.show()
        ax.cla()
    fig.clf()
    if len(df.columns) == 3:
        # plot grouped coefficients
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
        df_sigma = df.mul(sigma_x, axis=0)
        if exports:
            df_sigma = df_sigma / 1000
        df_sigma = df_sigma.rename(columns=EC_BY_YEARS)
        plot_coefficients_grouped(df=df_sigma, ax=ax)
        ax.tick_params(axis="x", labelsize=16)
        ax.set_ylabel(r"$c_{X} \cdot \sigma_{X} \;$ (" + unit + ")")
        fig.savefig(
            fig_dir + f"coefficients/coefficients_sigma_grouped.pdf",
            bbox_inches="tight",
        )
        plt.show()
        ax.cla()

    # plot all strucutre coefficients
    df = coefficients.drop("intercept")
    if exports:
        for col in df.columns:
            for idx in df.index:
                if "gas_price" in idx or "carbon_price" in idx:
                    df.loc[idx] /= 100  # convert to 100
    else:
        for col in df.columns:
            for idx in df.index:
                if (
                    "gas_price" not in idx
                    and "carbon_price" not in idx
                    and "net_export" not in idx
                ):
                    df.loc[idx] *= 1000  # convert to GW
    df = df.rename(index=VAR_NAMES_MAP)
    for col in df.columns:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
        plot_coefficients(df=df, col=col, ax=ax, color=color)
        ax.tick_params(axis="x", labelsize=16)
        ax.set_ylabel(r"$c_{X}$ (mixed units)")
        fig.savefig(
            fig_dir + f"coefficients/all_coefficients_{col}.pdf", bbox_inches="tight"
        )
        plt.show()
        ax.cla()
    fig.clf()
    if len(df.columns) == 3:
        # plot grouped coefficients
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
        df = df.rename(columns=EC_BY_YEARS)
        plot_coefficients_grouped(df=df, ax=ax)
        ax.tick_params(axis="x", labelsize=16)
        ax.set_ylabel(r"$c_{X}$ (mixed units)")
        fig.savefig(
            fig_dir + f"coefficients/all_coefficients_grouped.pdf",
            bbox_inches="tight",
        )
        plt.show()
        ax.cla()
