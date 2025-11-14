import pandas as pd
from src.train import clean_data

def test_clean_data_removes_nulls_and_invalids():
    df = pd.DataFrame({
        "col1": [1, 2, None],
        "col2": ["ok", " ?", "null"]
    })
    cleaned = clean_data(df)
    assert cleaned.isnull().sum().sum() == 0
    assert not cleaned.isin(["?", "error", "NAN", "invalid", "null"]).any().any()
