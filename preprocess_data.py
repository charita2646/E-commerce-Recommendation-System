import pandas as pd
import numpy as np

def clean_data(file_path):
    df = pd.read_csv(file_path)

    # Replace invalid values with NaN
    df.replace(["", " ", "NA", "null"], np.nan, inplace=True)

    # Drop unwanted column
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    # Convert User's ID to numeric
    df["User's ID"] = pd.to_numeric(df["User's ID"], errors="coerce")

    # Convert ProdID to numeric
    df["ProdID"] = pd.to_numeric(df["ProdID"], errors="coerce")

    # Remove rows with NaN in ID columns
    df.dropna(subset=["User's ID", "ProdID"], inplace=True)

    # Remove rows where ID or ProdID is 0
    df = df[(df["User's ID"] != 0) & (df["ProdID"] != 0)]

    # Fill text columns with empty string
    text_columns = df.select_dtypes(include="object").columns
    for col in text_columns:
        df[col] = df[col].fillna("")

    # Remove "|" from ImageURL
    if "ImageURL" in df.columns:
        df["ImageURL"] = df["ImageURL"].str.replace("|", "", regex=False)

    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df
