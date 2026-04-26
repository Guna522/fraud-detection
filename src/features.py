import pandas as pd

def add_features(df):
    df = df.copy()
    
    df["time_diff"] = 0
    df["amount_mean_last5"] = df["Amount"]
    df["amount_std_last5"] = 0
    df["tx_count_last10"] = 1

    return df