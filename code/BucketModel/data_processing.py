import pandas as pd
from BucketModel import BucketModel


def preprocess_for_bucket_model(processed_mat_df: pd.DataFrame) -> pd.DataFrame:
    """Processes the DataFrame from the processed_mat_file function to a format that can be used for the BucketModel.

    Args:
        processed_mat_df (pd.DataFrame): The DataFrame containing the data from the processed_mat_file function.

    Returns:
        pd.DataFrame | None: A DataFrame containing the data in a format that can be used for the BucketModel.
    """

    months = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }
    df = processed_mat_df.copy()

    df["Month"] = df["Month"].map(months)
    df["date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
    df = df.set_index("date")

    df["P_mix"] = df["Precipitation"]

    keep_columns = ["P_mix", "T_max", "T_min"]

    if "Simulation" in df.columns:
        keep_columns.append("Simulation")

    df = df[keep_columns]

    return df


def train_validate_split(data: pd.DataFrame, train_size: float) -> tuple:
    """Splits the data into training and validating sets.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        train_size (float): The proportion of the data to use for training. This is a value between 0 and 1.

    Returns:
        tuple: A tuple containing the training and testing DataFrames.
    """

    train_size = int(len(data) * train_size)
    train_data = data.iloc[:train_size]
    validate_data = data.iloc[train_size:]

    return train_data, validate_data


def run_multiple_simulations(
    preprocessed_simulated_data: pd.DataFrame,
    bucket_model: BucketModel,
    n_simulations: int,
) -> pd.DataFrame:
    """
    Run multiple simulations using the bucket model.

    Args:
        preprocessed_simulated_data (pd.DataFrame): Preprocessed simulation data with columns
            'P_mix', 'T_max', 'T_min', and 'Simulation'.
        bucket_model (BucketModel): Instance of the BucketModel.
        n_simulations (int): Number of simulations to run.

    Returns:
        pd.DataFrame: Results of all simulations.
    """
    results = pd.DataFrame(
        columns=[
            "ET",
            "Q_s",
            "Q_gw",
            "Snow_accum",
            "S",
            "S_gw",
            "Snow_melt",
            "Rain",
            "Snow",
            "Precip",
            "Simulation",
        ]
    )

    for simul in range(1, n_simulations + 1):
        # Create a fresh copy of the model for each simulation
        model_copy = bucket_model.copy()

        data = preprocessed_simulated_data[
            preprocessed_simulated_data["Simulation"] == simul
        ]

        # Verify required columns exist
        required_cols = ["P_mix", "T_max", "T_min"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data missing required columns: {required_cols}")

        results_simul = model_copy.run(data=data)
        results_simul["Simulation"] = simul

        results = pd.concat([results, results_simul])

    return results


def main() -> None:
    """
    This is an example of how you can use the preprocess_data function. You need to change the path_to_file and output_destination to your own paths.
    Alternatively you can import this function into another script and use it there. See example_run.ipynb for more information.
    """

    path_to_file = "/Users/cooper/Desktop/bucket-model/data/GSTEIGmeteo.txt"
    output_destination = "/Users/cooper/Desktop/bucket-model/data/GSTEIGmeteo.csv"
    catchment_area = 384.2  # km^2
    data = preprocess_for_bucket_model(path_to_file, output_destination, catchment_area)
    print(data)


if __name__ == "__main__":
    main()
