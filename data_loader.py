import os
import pandas as pd
import kagglehub

def load_data():
    # Download dataset automatically
    path = kagglehub.dataset_download("aaron7sun/stocknews")

    # Find CSV file
    csv_file = None
    for file in os.listdir(path):
        if file.endswith(".csv"):
            csv_file = os.path.join(path, file)
            break

    if csv_file is None:
        raise FileNotFoundError("CSV file not found in dataset!")

    df = pd.read_csv(csv_file)

    # Combine Top1–Top25 into one text column
    headline_cols = [f"Top{i}" for i in range(1, 26)]
    df["combined_news"] = df[headline_cols].fillna("").agg(" ".join, axis=1)

    # Keep only necessary columns
    df = df[["combined_news", "Label"]]

    return df