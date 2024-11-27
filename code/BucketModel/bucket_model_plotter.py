import seaborn as sns  # For styling the plots
import matplotlib.pyplot as plt  # For plotting
from matplotlib.gridspec import GridSpec  # For creating subplots

from scipy.stats import gaussian_kde  # For the density plot
import numpy as np  # For the density plot

import scipy.stats as stats  # For the confidence interval

import pandas as pd  # For the data handling


def calculate_total_runoff(results: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate total runoff by summing surface runoff (Q_s) and groundwater runoff (Q_gw).

    Args:
        results (pd.DataFrame): The results dataframe containing Q_s and Q_gw columns.

    Returns:
        pd.DataFrame: The input dataframe with an additional 'Total_Runoff' column.
    """
    results_copy = results.copy()
    results_copy["Total_Runoff"] = results_copy["Q_s"] + results_copy["Q_gw"]
    return results_copy


def filter_data_by_date(
    data: pd.DataFrame, start_year: str, end_year: str
) -> pd.DataFrame:
    """
    Filter the dataframe to include only data within the specified date range.

    Args:
        data (pd.DataFrame): The input dataframe with a datetime index.
        start_year (str): The start year of the date range (inclusive).
        end_year (str): The end year of the date range (inclusive).

    Returns:
        pd.DataFrame: The filtered dataframe.
    """
    return data[start_year:end_year]


def plot_water_balance(
    results: pd.DataFrame,
    title: str = "",
    output_destination: str = "",
    palette: list = ["#004E64", "#007A9A", "#00A5CF", "#9FFFCB", "#25A18E"],
    start_year: str = "1986",
    end_year: str = "2000",
    figsize: tuple[int, int] = (10, 6),
    fontsize: int = 12,
) -> None:
    """Plot the water balance of the model.

    Args:
        results (pd.DataFrame): The results from the model run.
        title (str): The title of the plot, if empty, no title will be shown.
        output_destination (str): The path to the output file, if empty, the plot will not be saved.
        palette (list): The color palette to use for the plot, default is ['#004E64', '#007A9A', '#00A5CF', '#9FFFCB', '#25A18E'].
        start_year (str): The start year of the plot, default is '1986'.
        end_year (str): The end year of the plot, default is '2000'.
        figsize (tuple): The size of the figure, default is (10, 6).
        fontsize (int): The fontsize of the plot, default is 12.
    """
    # Some style settings
    BAR_WIDTH = 0.35
    sns.set_context("paper")
    sns.set_style("white")

    # Helper function to plot a single bar chart layer
    def plot_bar_layer(
        ax: plt.Axes,
        positions: np.ndarray,
        heights: np.ndarray,
        label: str,
        color: str,
        bottom_layer_heights: np.ndarray = None,
    ) -> None:
        """Plot a single layer of a bar chart.

        Args:
            ax (plt.Axes): The axis to plot on.
            positions (np.ndarray): The x-positions of the bars.
            heights (np.ndarray): The heights of the bars.
            label (str): The label of the layer.
            color (str): The color of the layer.
            bottom_layer_heights (np.ndarray): The heights of the bottom layer, default is None.
        """
        ax.bar(
            positions,
            heights,
            width=BAR_WIDTH,
            label=label,
            color=color,
            bottom=bottom_layer_heights,
        )

    # Prepare the data
    results_filtered = results.copy()
    results_filtered["Year"] = results_filtered.index.year
    results_filtered = results_filtered[start_year:end_year]
    yearly_totals = results_filtered.groupby("Year").sum()

    years = yearly_totals.index

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each component of the water balance
    plot_bar_layer(ax, years - BAR_WIDTH / 2, yearly_totals["Rain"], "Rain", palette[0])
    plot_bar_layer(
        ax,
        years - BAR_WIDTH / 2,
        yearly_totals["Snow"],
        "Snow",
        palette[1],
        bottom_layer_heights=yearly_totals["Rain"],
    )
    plot_bar_layer(
        ax, years + BAR_WIDTH / 2, yearly_totals["Q_s"], "Q$_{surface}$", palette[2]
    )
    plot_bar_layer(
        ax,
        years + BAR_WIDTH / 2,
        yearly_totals["Q_gw"],
        "Q$_{gw}$",
        palette[3],
        bottom_layer_heights=yearly_totals["Q_s"],
    )
    plot_bar_layer(
        ax,
        years + BAR_WIDTH / 2,
        yearly_totals["ET"],
        "ET",
        palette[4],
        bottom_layer_heights=yearly_totals["Q_s"] + yearly_totals["Q_gw"],
    )

    ax.tick_params(which="both", length=10, width=2, labelsize=fontsize)
    ax.set_ylabel("Water depth [mm]", fontsize=fontsize)
    ax.legend(fontsize=fontsize, ncol=3, loc="best")
    plt.tight_layout()
    sns.despine()

    if title:
        plt.title(title)

    if output_destination:
        fig.savefig(output_destination, dpi=300, bbox_inches="tight")


def plot_Q_Q(
    results: pd.DataFrame,
    observed: pd.DataFrame,
    title: str = "",
    output_destination: str = "",
    color: str = "#007A9A",
    figsize: tuple[int, int] = (6, 6),
    fontsize: int = 12,
    line: bool = True,
    kde: bool = True,
    cmap: str = "rainbow",
) -> None:
    """Plot the observed vs simulated total runoff (Q) values.

    Args:
        results (pd.DataFrame): The results from the model run.
        observed (pd.DataFrame): The observed data. Should contain the column 'Q' for the observed runoff.
        title (str): The title of the plot, if empty, no title will be shown.
        output_destination (str): The path to the output file, if empty, the plot will not be saved.
        color (str): The color of the plot, default is '#007A9A'.
        figsize (tuple): The size of the figure, default is (6, 6).
        fontsize (int): The fontsize of the plot, default is 12.
        line (bool): If True, a 1:1 line will be plotted, default is True.
        kde (bool): If True, a kernel density estimate will be plotted, default is True.
        cmap (str): The colormap to use for the kde, default is 'rainbow'.
    """
    sns.set_context("paper")

    # Prepare the data
    results_filtered = calculate_total_runoff(results)

    fig, ax = plt.subplots(figsize=figsize)

    if kde:
        xy = np.vstack([results_filtered["Total_Runoff"], observed["Q"]])
        z = gaussian_kde(xy)(xy)
        sns.scatterplot(
            x=results_filtered["Total_Runoff"],
            y=observed["Q"],
            ax=ax,
            c=z,
            s=30,
            cmap=cmap,
            edgecolor="none",
        )
    else:
        sns.scatterplot(
            x=results_filtered["Total_Runoff"],
            y=observed["Q"],
            ax=ax,
            color=color,
            s=30,
            edgecolor="none",
        )

    if line:
        min_value = min(results_filtered["Total_Runoff"].min(), observed["Q"].min())
        max_value = max(results_filtered["Total_Runoff"].max(), observed["Q"].max())
        ax.plot(
            [min_value, max_value],
            [min_value, max_value],
            color="black",
            linestyle="--",
        )

    ax.set_xlabel("Simulated total runoff [mm/d]", fontsize=fontsize)
    ax.set_ylabel("Observed total runoff [mm/d]", fontsize=fontsize)
    ax.tick_params(which="both", length=10, width=2, labelsize=fontsize)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    sns.despine()

    min_x = results_filtered["Total_Runoff"].min()
    max_x = results_filtered["Total_Runoff"].max()
    min_y = observed["Q"].min()
    max_y = observed["Q"].max()

    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)

    if title:
        plt.title(title)

    if output_destination:
        fig.savefig(output_destination, dpi=300, bbox_inches="tight")


def plot_monthly_boxplot(
    results: pd.DataFrame,
    title: str = "",
    output_destination: str = "",
    figsize: tuple[int, int] = (12, 12),
    fontsize: int = 12,
    palette: list = ["#004E64", "#007A9A", "#00A5CF", "#9FFFCB"],
) -> None:
    """Plot the monthly boxplot of the simulated environmental variables.

    Args:
        results (pd.DataFrame): The results from the model run, make sure you have the following columns: 'Precip', 'ET', 'Snow_melt', 'Q_s', 'Q_gw'.
        title (str): The title of the plot, if empty, no title will be shown.
        output_destination (str): The path to the output file, if empty, the plot will not be saved.
        figsize (tuple): The size of the figure, default is (12, 12).
        fontsize (int): The fontsize of the plot, default is 12.
        palette (list): The color palette to use for the plot, default is ['#004E64', '#007A9A', '#00A5CF', '#9FFFCB']. The first color is for precipitation, the second color is for ET, the third color is for snowmelt, the fourth color is for total runoff.
    """
    sns.set_context("paper")

    # Prepare the data
    results_filtered = calculate_total_runoff(results)
    results_filtered["Month"] = results_filtered.index.month
    results_filtered["Year"] = results_filtered.index.year

    monthly_sums = results_filtered.groupby(["Year", "Month"]).sum().reset_index()

    months = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }

    monthly_sums["Month"] = monthly_sums["Month"].map(months)

    fig = plt.figure(figsize=figsize)
    layout = (2, 2)

    ax_precip = plt.subplot2grid(layout, (0, 0))
    ax_et = plt.subplot2grid(layout, (0, 1))
    ax_snow_melt = plt.subplot2grid(layout, (1, 0))
    ax_runoff = plt.subplot2grid(layout, (1, 1))

    sns.boxplot(
        x="Month", y="Precip", data=monthly_sums, ax=ax_precip, color=palette[0]
    )
    sns.boxplot(x="Month", y="ET", data=monthly_sums, ax=ax_et, color=palette[1])
    sns.boxplot(
        x="Month", y="Snow_melt", data=monthly_sums, ax=ax_snow_melt, color=palette[2]
    )
    sns.boxplot(
        x="Month", y="Total_Runoff", data=monthly_sums, ax=ax_runoff, color=palette[3]
    )

    ax_precip.set_xlabel("")
    ax_precip.set_ylabel("Precipitation [mm/d]", fontsize=fontsize)
    ax_precip.tick_params(which="both", length=10, width=2, labelsize=fontsize)
    ax_precip.set_title("Monthly Precipitation", fontsize=fontsize)
    ax_precip.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    ax_et.set_xlabel("")
    ax_et.set_ylabel("Actual ET [mm/d]", fontsize=fontsize)
    ax_et.tick_params(which="both", length=10, width=2, labelsize=fontsize)
    ax_et.set_title("Monthly Actual ET", fontsize=fontsize)
    ax_et.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    ax_snow_melt.set_xlabel("")
    ax_snow_melt.set_ylabel("Snowmelt [mm/d]", fontsize=fontsize)
    ax_snow_melt.tick_params(which="both", length=10, width=2, labelsize=fontsize)
    ax_snow_melt.set_title("Monthly Snowmelt", fontsize=fontsize)
    ax_snow_melt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    ax_runoff.set_xlabel("")
    ax_runoff.set_ylabel("Total Runoff [mm/d]", fontsize=fontsize)
    ax_runoff.tick_params(which="both", length=10, width=2, labelsize=fontsize)
    ax_runoff.set_title("Monthly Total Runoff", fontsize=fontsize)
    ax_runoff.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    sns.despine()

    if title:
        plt.suptitle(title, fontsize=fontsize)

    if output_destination:
        fig.savefig(output_destination, dpi=300, bbox_inches="tight")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec


def prepare_data(
    results: pd.DataFrame,
    start_year: str,
    end_year: str,
    monthly: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare the data for plotting by filtering it based on the specified date range.

    Args:
        results (pd.DataFrame): The results dataframe.
        start_year (str): The start year of the date range (inclusive).
        end_year (str): The end year of the date range (inclusive).
        monthly (bool): If True, the data will be resampled to monthly values, default is False.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The filtered results and observed dataframes.
    """
    results_filtered = filter_data_by_date(results, start_year, end_year)
    results_filtered = calculate_total_runoff(results_filtered)

    if monthly:
        results_filtered = results_filtered.resample("M").sum()

    return results_filtered


def plot_runoff(
    ax: plt.Axes,
    results_filtered: pd.DataFrame,
    palette: list[str],
    fontsize: int,
    monthly: bool,
) -> plt.Line2D:
    """
    Plot the simulated total runoff (Q) values.

    Args:
        ax (plt.Axes): The axis to plot on.
        results_filtered (pd.DataFrame): The filtered results dataframe.
        palette (list): The color palette to use for the plot.
        fontsize (int): The fontsize of the plot.
        monthly (bool): If True, the data will be resampled to monthly values.

    Returns:
        plt.Line2D: The line object representing the plot."""
    (line,) = ax.plot(
        results_filtered.index,
        results_filtered["Total_Runoff"],
        color=palette[0],
        label="Simulated total runoff",
        alpha=0.7,
    )

    ax.set_xlabel("", fontsize=fontsize)
    ylabel = "Total runoff [mm/month]" if monthly else "Total runoff [mm/d]"
    ax.set_ylabel(ylabel, fontsize=fontsize)

    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    return line


def plot_precipitation_ts(
    ax: plt.Axes,
    results_filtered: pd.DataFrame,
    palette: list[str],
    fontsize: int,
    monthly: bool,
) -> plt.Line2D:
    """
    Plot the precipitation values.

    Args:
        ax (plt.Axes): The axis to plot on.
        results_filtered (pd.DataFrame): The filtered results dataframe.
        palette (list): The color palette to use for the plot.
        fontsize (int): The fontsize of the plot.
        monthly (bool): If True, the data will be resampled to monthly values.

    Returns:
        plt.Line2D: The line object representing the plot."""
    precip = (
        results_filtered["Precip"].resample("M").sum()
        if monthly
        else results_filtered["Precip"]
    )

    (precip_line,) = ax.plot(
        precip.index,
        precip,
        color=palette[1],
        label="Precipitation",
        linewidth=1,
    )

    ax.set_ylabel("Prcp [mm/month]" if monthly else "Prcp [mm/d]", fontsize=fontsize)
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.set_ylim(0, precip.max() * 1.5)
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    return precip_line


def plot_timeseries(
    results: pd.DataFrame,
    start_year: str,
    end_year: str,
    monthly: bool = False,
    plot_precipitation: bool = False,
    title: str = "",
    output_destination: str = "",
    figsize: tuple[int, int] = (10, 8),
    fontsize: int = 12,
    palette: list = ["#007A9A", "#8B4513"],
) -> None:
    """
    Plot the timeseries of the simulated total runoff (Q) values,
    with an option to include precipitation in a smaller subplot above the main plot.

    Args:
        results (pd.DataFrame): The results from the model run.
        start_year (str): The start year of the date range (inclusive).
        end_year (str): The end year of the date range (inclusive).
        monthly (bool): If True, the data will be resampled to monthly values, default is False.
        plot_precipitation (bool): If True, a subplot with precipitation will be included, default is False.
        title (str): The title of the plot, if empty, no title will be shown.
        output_destination (str): The path to the output file, if empty, the plot will not be saved.
        figsize (tuple): The size of the figure, default is (10, 8).
        fontsize (int): The fontsize of the plot, default is 12.
        palette (list): The color palette to use for the plot, default is ['#007A9A', '#8B4513']. The first color is for runoff, the second color is for precipitation.
    """
    sns.set_context("paper")

    # Prepare the data
    results_filtered = prepare_data(results, start_year, end_year, monthly)

    # Create figure and grid layout
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = GridSpec(2, 1, height_ratios=[1, 4], hspace=0.15)

    # Main plot (runoff)
    ax1 = fig.add_subplot(gs[1])
    line = plot_runoff(ax1, results_filtered, palette, fontsize, monthly)

    # Precipitation plot
    if plot_precipitation:
        ax2 = fig.add_subplot(gs[0], sharex=ax1)
        precip_line = plot_precipitation_ts(
            ax2, results_filtered, palette, fontsize, monthly
        )
    else:
        precip_line = None

    # Set title
    plt.suptitle(title, fontsize=fontsize + 2)

    # Create legend
    handles = [line] + ([precip_line] if precip_line else [])
    ax1.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=len(handles),
        fontsize=fontsize,
    )

    sns.despine()

    # Save the plot if output destination is provided
    if output_destination:
        fig.savefig(output_destination, dpi=300, bbox_inches="tight")

    plt.show()


def plot_parameter_kde(
    n_fold_results: pd.DataFrame,
    bounds: dict,
    output_destination: str = None,
    figsize: tuple[int, int] = (10, 6),
    fontsize: int = 12,
    plot_type: str = "histplot",
) -> None:
    """Plot the histogram of the parameters.

    Args:
        n_fold_results (pd.DataFrame): The n_fold_results from the model calibration.
        bounds (dict): The bounds of the parameters. They are used to set the x-axis limits.
        output_destination (str): The path to the output file.
        figsize (tuple): The size of the figure, default is (10, 6).
        fontsize (int): The fontsize of the plot, default is 12.
        plot_type (str): The type of plot to use. Can be either 'histplot' or 'kdeplot', default is 'histplot'.
    """
    sns.set_context("paper")

    # Prepare the data
    n_fold_results_filtered = n_fold_results.copy()

    fig = plt.figure(figsize=figsize)
    layout = (2, 3)

    ax_k = plt.subplot2grid(layout, (0, 0))
    ax_S_max = plt.subplot2grid(layout, (0, 1))
    ax_fr = plt.subplot2grid(layout, (0, 2))
    ax_rg = plt.subplot2grid(layout, (1, 0))
    ax_gauge_adj = plt.subplot2grid(layout, (1, 1))

    if plot_type == "histplot":
        bins = int(np.sqrt(len(n_fold_results_filtered)))
        sns.histplot(
            data=n_fold_results_filtered["k"], ax=ax_k, color="#007A9A", bins=bins
        )
        sns.histplot(
            data=n_fold_results_filtered["S_max"],
            ax=ax_S_max,
            color="#007A9A",
            bins=bins,
        )
        sns.histplot(
            data=n_fold_results_filtered["fr"], ax=ax_fr, color="#007A9A", bins=bins
        )
        sns.histplot(
            data=n_fold_results_filtered["rg"], ax=ax_rg, color="#007A9A", bins=bins
        )
        sns.histplot(
            data=n_fold_results_filtered["gauge_adj"],
            ax=ax_gauge_adj,
            color="#007A9A",
            bins=bins,
        )
    elif plot_type == "kdeplot":
        sns.kdeplot(
            data=n_fold_results_filtered["k"], ax=ax_k, color="#007A9A", fill=True
        )
        sns.kdeplot(
            data=n_fold_results_filtered["S_max"],
            ax=ax_S_max,
            color="#007A9A",
            fill=True,
        )
        sns.kdeplot(
            data=n_fold_results_filtered["fr"], ax=ax_fr, color="#007A9A", fill=True
        )
        sns.kdeplot(
            data=n_fold_results_filtered["rg"], ax=ax_rg, color="#007A9A", fill=True
        )
        sns.kdeplot(
            data=n_fold_results_filtered["gauge_adj"],
            ax=ax_gauge_adj,
            color="#007A9A",
            fill=True,
        )

    ax_k.set_xlabel("k", fontsize=fontsize)
    ax_k.set_ylabel("Density", fontsize=fontsize)
    ax_k.tick_params(which="both", length=10, width=2, labelsize=fontsize)
    ax_k.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax_k.set_xlim(bounds["k"])

    ax_S_max.set_xlabel("S$_{max}$", fontsize=fontsize)
    ax_S_max.set_ylabel("Density", fontsize=fontsize)
    ax_S_max.tick_params(which="both", length=10, width=2, labelsize=fontsize)
    ax_S_max.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax_S_max.set_xlim(bounds["S_max"])

    ax_fr.set_xlabel("fr", fontsize=fontsize)
    ax_fr.set_ylabel("Density", fontsize=fontsize)
    ax_fr.tick_params(which="both", length=10, width=2, labelsize=fontsize)
    ax_fr.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax_fr.set_xlim(bounds["fr"])

    ax_rg.set_xlabel("rg", fontsize=fontsize)
    ax_rg.set_ylabel("Density", fontsize=fontsize)
    ax_rg.tick_params(which="both", length=10, width=2, labelsize=fontsize)
    ax_rg.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax_rg.set_xlim(bounds["rg"])

    ax_gauge_adj.set_xlabel("gauge_adj", fontsize=fontsize)
    ax_gauge_adj.set_ylabel("Density", fontsize=fontsize)
    ax_gauge_adj.tick_params(which="both", length=10, width=2, labelsize=fontsize)
    ax_gauge_adj.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax_gauge_adj.set_xlim(bounds["gauge_adj"])

    plt.tight_layout()
    sns.despine()

    if output_destination:
        fig.savefig(output_destination, dpi=300, bbox_inches="tight")


def group_by_month_with_ci(
    results_df: pd.DataFrame, n_simulations: int = 50, confidence_level: float = 0.95
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate the mean and confidence interval of monthly summed data.

    First sums values for each month within years/simulations, then
    calculates the mean and confidence intervals of these monthly sums
    across all years or simulations.

    Args:
        results_df (pd.DataFrame): The results from running the BucketModel for multiple simulations.
        n_simulations (int): The number of simulations in the generated ensemble, default is 50.
        confidence_level (float): Desired confidence level (default 0.95 for 95% CI)

    Returns:
        monthly_mean (pd.DataFrame): Mean of monthly sums across years/simulations.
        ci (pd.DataFrame): Confidence interval of monthly sums.
    """
    results_df = results_df.copy()
    results_df["month"] = results_df.index.month
    results_df["year"] = results_df.index.year

    # Group by simulation if available
    if "Simulation" in results_df.columns:
        monthly_data = (
            results_df.groupby(["Simulation", "year", "month"]).sum().reset_index()
        )
        # Calculate degrees of freedom based on number of simulations
        df = n_simulations - 1
    else:
        monthly_data = results_df.groupby(["year", "month"]).sum().reset_index()
        # Calculate degrees of freedom based on number of years
        df = len(monthly_data["year"].unique()) - 1

    monthly_mean = monthly_data.groupby("month").mean()
    monthly_std = monthly_data.groupby("month").std()

    # Calculate alpha for the desired confidence level
    alpha = 1 - confidence_level

    # Calculate t-value for two-tailed test
    t_value = stats.t.ppf(1 - alpha / 2, df)

    # Calculate standard error
    if "Simulation" in results_df.columns:
        n = n_simulations
    else:
        n = len(monthly_data["year"].unique())

    standard_error = monthly_std / np.sqrt(n)

    # Calculate confidence intervals
    ci = t_value * standard_error

   # monthly_mean = monthly_mean.drop(columns=["Simulation"])

    return monthly_mean, ci


def plot_monthly_runoff_with_ci(
    results_monthly: pd.DataFrame, ci: pd.DataFrame, output_destination: str = ""
) -> None:
    """
    Plots mean monthly total runoff with 95% confidence interval and saves the plot.

    Args:
        results_monthly (pd.DataFrame): DataFrame containing the monthly results with mean values.
        ci (pd.DataFrame): DataFrame containing the confidence intervals for each month.
        output_destination (str): File path to save the plot.
    """
    results_monthly["total_runoff"] = results_monthly["Q_s"] + results_monthly["Q_gw"]
    ci_total_runoff = ci["Q_s"] + ci["Q_gw"]

    plt.figure(figsize=(10, 6))
    plt.plot(
        results_monthly.index,
        results_monthly["total_runoff"],
        label="Total Runoff [mm/month]",  # Updated unit
        color="b",
    )
    plt.fill_between(
        results_monthly.index,
        results_monthly["total_runoff"] - ci_total_runoff,
        results_monthly["total_runoff"] + ci_total_runoff,
        color="b",
        alpha=0.2,
        label="95% CI",
    )
    plt.xlabel("Month")
    plt.ylabel("Runoff [mm/month]")
    plt.title("Mean Monthly Total Runoff with 95% Confidence Interval")
    plt.legend()
    sns.despine()
    plt.grid(linestyle="-", alpha=0.7)
