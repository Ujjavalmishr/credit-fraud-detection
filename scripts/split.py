# scripts/split.py
from typing import Dict
import pandas as pd

def time_split(df: pd.DataFrame, test_frac: float = 0.2) -> Dict[str, pd.DataFrame]:
    df = df.sort_values("Time").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_frac))
    return {
        "train": df.iloc[:split_idx].copy(),
        "test":  df.iloc[split_idx:].copy()
    }

# Example usage:
df = pd.read_csv("data/creditcard.csv")
splits = time_split(df)
train, test = splits["train"], splits["test"]

print(train.shape, test.shape)
