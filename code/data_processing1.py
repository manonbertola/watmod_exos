import pandas as pd
import scipy.io
import os
import numpy as np

# Define the custom order of months
MONTH_ORDER = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


def _map_numeric_to_month(month_numeric: int) -> str:
    """
    Map a numeric month to its corresponding string representation.

    Args:
        month_numeric (int): The numeric representation of the month.

    Returns:
        str: The string representation of the month or 'Invalid Month'.
    """
    if 1 <= month_numeric <= 12:
        return MONTH_ORDER[month_numeric - 1]
    return "Invalid Month"


def _is_input_data_mat_file(file_path: str) -> bool:
    """
    Check if the file is an input data file.

    Args:
        file_path (str): The path of the file.

    Returns:
        bool: True if the file is an input data file, False if it's a simulation file.
    """
    mat_data = scipy.io.loadmat(file_path)
    # Check for the presence of yearP which is unique to the input format
    return "yearP" in mat_data


def create_date_index(year: int, num_days: int = 365) -> pd.DatetimeIndex:
    """
    Create a DatetimeIndex for a specific year.

    Args:
        year (int): The year to create the index for
        num_days (int): Number of days to generate (default: 365)

    Returns:
        pd.DatetimeIndex: DatetimeIndex for the specified year
    """
    start_date = pd.to_datetime(f"{year}-01-01")
    return pd.date_range(start_date, periods=num_days)


def process_input_data(mat_data: dict) -> pd.DataFrame:
    """
    Process input data format (.mat file with P, Tmax, Tmin, yearP structure).

    Args:
        mat_data (dict): The loaded .mat file data

    Returns:
        pd.DataFrame: Processed DataFrame
    """
    # Extract the data arrays
    P = mat_data["P"]  # Shape: (30, 365)
    Tmax = mat_data["Tmax"]  # Shape: (30, 365)
    Tmin = mat_data["Tmin"]  # Shape: (30, 365)
    years = mat_data["yearP"].flatten()  # Shape: (30,)

    # Initialize an empty list to store DataFrames for each year
    yearly_dfs = []

    # Process each year's data
    for year_idx, year in enumerate(years):
        # Create daily dates for the current year
        dates = create_date_index(year)

        # Create DataFrame for current year
        year_df = pd.DataFrame(
            {
                "Precipitation": P[year_idx],
                "T_max": Tmax[year_idx],
                "T_min": Tmin[year_idx],
                "Date": dates,
            }
        )

        # Calculate average temperature
        year_df["T_avg"] = (year_df["T_max"] + year_df["T_min"]) / 2

        # Add year information
        year_df["Year"] = year

        yearly_dfs.append(year_df)

    # Combine all years into a single DataFrame
    return pd.concat(yearly_dfs, ignore_index=True)


def process_simulation_data(mat_data: dict) -> pd.DataFrame:
    """
    Process simulation data format (.mat file with gP, gTmax, gTmin structure).

    Args:
        mat_data (dict): The loaded .mat file data

    Returns:
        pd.DataFrame: Processed DataFrame
    """
    # Extract the data arrays
    gP = mat_data["gP"]  # Shape: (1500, 365)
    gTmax = mat_data["gTmax"]  # Shape: (1500, 365)
    gTmin = mat_data["gTmin"]  # Shape: (1500, 365)

    # Calculate number of simulations (assuming 30 years per simulation)
    years_per_simulation = 30
    num_simulations = gP.shape[0] // years_per_simulation

    # Initialize an empty list to store DataFrames
    all_dfs = []

    # Process each simulation
    for sim_num in range(num_simulations):
        start_idx = sim_num * years_per_simulation
        end_idx = start_idx + years_per_simulation

        # Extract data for current simulation
        sim_P = gP[start_idx:end_idx]
        sim_Tmax = gTmax[start_idx:end_idx]
        sim_Tmin = gTmin[start_idx:end_idx]

        # Process each year in the simulation
        for year_idx in range(years_per_simulation):
            # Create dates for current year (starting from 1980)
            year = 1980 + year_idx
            dates = create_date_index(year)

            # Create DataFrame for current year
            year_df = pd.DataFrame(
                {
                    "Precipitation": sim_P[year_idx],
                    "T_max": sim_Tmax[year_idx],
                    "T_min": sim_Tmin[year_idx],
                    "Date": dates,
                    "Year": year,
                    "Simulation": sim_num + 1,  # Moved Simulation to the end
                }
            )

            # Calculate average temperature
            year_df["T_avg"] = (year_df["T_max"] + year_df["T_min"]) / 2

            all_dfs.append(year_df)

    # Combine all DataFrames
    df = pd.concat(all_dfs, ignore_index=True)

    return df


def handle_leap_years(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust dates in the DataFrame for leap years to align dates from March 1st onwards with non-leap years.

    Args:
        df (pd.DataFrame): DataFrame containing a 'Date' column in datetime format.

    Returns:
        pd.DataFrame: Modified DataFrame with adjusted dates for leap years.
    """
    leap_years = df["Date"].dt.year[df["Date"].dt.is_leap_year].unique()
    for year in leap_years:
        start_date = pd.to_datetime(f"{year}-03-01")
        mask = (df["Date"].dt.year == year) & (df["Date"].dt.month >= 3)
        df.loc[mask, "Date"] += pd.DateOffset(days=1)
    return df


def skip_feb_29(df: pd.DataFrame) -> pd.DataFrame:
    """
    Skip February 29 in the Date column of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to transform.

    Returns:
        pd.DataFrame: The DataFrame with February 29 skipped.
    """
    df.loc[
        (df["Date"].dt.month == 2) & (df["Date"].dt.day == 29), "Date"
    ] += pd.DateOffset(days=1)
    return df


def add_month_day_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Month and Day columns to the DataFrame based on the Date column.

    Args:
        df (pd.DataFrame): The DataFrame to transform.

    Returns:
        pd.DataFrame: The DataFrame with the added Month and Day columns.
    """
    df["Month"] = df["Date"].dt.month.apply(_map_numeric_to_month)
    df["Month"] = pd.Categorical(df["Month"], categories=MONTH_ORDER, ordered=True)
    df["Day"] = df["Date"].dt.day
    df.drop(columns=["Date"], inplace=True)
    return df


def process_mat_file(file_path: str) -> pd.DataFrame:
    """
    Process the .mat file and return its content as a DataFrame.
    Handles both input data and simulation data formats.

    Args:
        file_path (str): The path of the .mat file.

    Returns:
        pd.DataFrame: The processed DataFrame with consistent column structure.
    """
    mat_data = scipy.io.loadmat(file_path)

    # Determine the type of data and process accordingly
    is_input = _is_input_data_mat_file(file_path)

    # Process the data based on its type
    if is_input:
        df = process_input_data(mat_data)
    else:
        df = process_simulation_data(mat_data)

    # Apply common transformations
    df = df.pipe(handle_leap_years).pipe(skip_feb_29).pipe(add_month_day_columns)

    return df
