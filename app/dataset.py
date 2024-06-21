from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split

import pandas as pd

def get_dataset(name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataset_path = Path.cwd() / 'datasets'
    dataset_name = name.lower().replace(' ', '_')
    X = pd.read_csv(dataset_path / f'{dataset_name}_X.csv')
    y = pd.read_csv(dataset_path / f'{dataset_name}_y.csv')
    return split_dataset(X, y)

def split_dataset(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)    
    return X_train, X_test, y_train, y_test
