import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genextreme, ks_2samp
from scipy.optimize import curve_fit
from dataclasses import dataclass, field
import seaborn as sns
from typing import Tuple, Optional, List, Union, Dict


@dataclass
class ClimateExtreme:
    """A class to analyze and compare extreme values in climate data.

    This class provides methods for fitting extreme value distributions,
    performing statistical tests, and visualizing comparisons between datasets.

    Attributes:
        data (pd.DataFrame): The input climate data.
        extreme (np.ndarray): Extracted extreme values.

    Methods:
        fit_genextreme: Fit a Generalized Extreme Value distribution.
        plot_fit_and_ci: Plot the fitted distribution with confidence intervals.
        truncated_ks_test: Perform a Kolmogorov-Smirnov test on extreme values.
        plot_extreme_comparison: Plot a comparison of extreme value distributions.
    """

    data: pd.DataFrame
    extreme: np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        # Create a deep copy of the data to prevent modifying the original
        self.data = self.data.copy()

    def truncated_ks_test(
        self, column: str, other: "ClimateExtreme", quantile: float
    ) -> Tuple[float, float]:
        # Calculate the threshold for extreme values based on the specified quantile
        threshold = self.data[column].quantile(quantile)

        # Extract the extreme values for both datasets
        observed_extreme = self.data[self.data[column] >= threshold][column]
        simulated_extreme = other.data[other.data[column] >= threshold][column]

        # Perform the KS test
        ks_stat, p_value = ks_2samp(observed_extreme, simulated_extreme)
        return ks_stat, p_value

    def run_simulation_ks_tests(
        self,
        observed_data: "ClimateExtreme",
        column: str = "Precipitation",
        quantile: float = 0.99,
    ) -> List[Tuple[float, float]]:
        """
        Run KS tests for extreme values in each simulation against observed data.

        Args:
            observed_data (ClimateExtreme): The observed data for comparison.
            column (str): The column to compare.
            quantile (float): The quantile threshold for defining extreme values.

        Returns:
            List[Tuple[float, float]]: A list of KS test results for each simulation.
        """

        simulations = self.data["Simulation"].unique()

        ks_results = []
        for sim in simulations:
            sim_data = self.data[self.data["Simulation"] == sim]

            sim_climate_extreme = ClimateExtreme(sim_data)
            ks_stat, p_value = observed_data.truncated_ks_test(
                column, sim_climate_extreme, quantile=quantile
            )

            ks_results.append((ks_stat, p_value))

        print("KS Tests Completed.")
        return ks_results

    def plot_ks_results(
        self,
        ks_results: List[Tuple[float, float]],
        significance_threshold: float = 0.05,
    ):
        """
        Plots the p-value distribution from KS tests with a customizable significance threshold.

        Args:
            ks_results (List[Tuple[float, float]]): The KS test results, where each tuple contains (KS statistic, p-value).
            significance_threshold (float): The p-value threshold for significance.
        """
        # Extract p-values from the KS test results
        p_values = [result[1] for result in ks_results]

        # Color based on significance threshold
        significance_colors = [
            "green" if p >= significance_threshold else "red" for p in p_values
        ]

        # Create the plot
        plt.figure(figsize=(8, 6))

        # Histogram of p-values
        plt.hist(p_values, bins=10, color="skyblue", edgecolor="black", alpha=0.7)
        plt.axvline(
            significance_threshold,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"{significance_threshold} Threshold",
        )
        plt.title("P-value Distribution Across Simulations")
        plt.xlabel("P-value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)

        # Overlay individual p-values with color-coding for significance
        for i, p in enumerate(p_values):
            plt.plot(
                p, 0, "o", color=significance_colors[i], markersize=8
            )  # Scatter points along the x-axis

        plt.tight_layout()
        sns.despine()
        plt.show()

    def plot_extreme_comparison(
        self,
        column: str,
        units: str,
        other: "ClimateExtreme",
        quantile: float = 0.95,
        output_destination: Optional[str] = None,
    ) -> None:
        """
        Plot the extreme value distributions of two datasets for comparison.

        Args:
            column (str): The column to compare.
            other (ClimateExtreme): Another ClimateExtreme object to compare against.
            quantile (float): The quantile threshold for defining extreme values.
            output_destination (Optional[str]): File path to save the figure. If None, the plot will be displayed.

        Raises:
            ValueError: If the column is not found in one or both datasets.
        """
        sns.set_context("paper", font_scale=1.4)

        threshold_self = self.data[column].quantile(quantile)
        threshold_other = other.data[column].quantile(quantile)

        extremes_self = self.data[self.data[column] > threshold_self][column].values
        extremes_other = other.data[other.data[column] > threshold_other][column].values

        plt.figure(figsize=(10, 6))
        sns.kdeplot(extremes_self, label="Generated Data", shade=True)
        sns.kdeplot(extremes_other, label="Observed Data", shade=True)

        plt.title(f"Comparison of Extreme Values ({quantile:.2%} quantile)")
        plt.xlabel(column + f" ({units})")
        plt.ylabel("Density")
        plt.xlim(20,120)
        sns.despine()
        plt.legend()

        if output_destination:
            plt.savefig(output_destination, bbox_inches="tight", dpi=300)
        else:
            plt.show()
