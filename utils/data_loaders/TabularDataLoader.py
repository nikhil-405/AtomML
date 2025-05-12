import numpy as np
import pandas as pd

def load_tabular_data(file_path, x = "x", y = "y"):
    df = pd.read_csv(file_path)
    x = df[x].values
    y = df[y].values
    return x, y