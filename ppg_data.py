from typing import Tuple
import pandas as pd
import numpy as np

def load_ppg_raw_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_excel(file_path)
    labels, data = [], []
    for i in df:
        labels.append(df[i][0]) # First row is the label
        data.append(df[i][1:]) # Second row onwards is the data
    return np.array(labels), np.array(data)
