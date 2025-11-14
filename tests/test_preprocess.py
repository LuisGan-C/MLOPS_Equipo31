import pandas as pd
from src.mlops_equipo31.train import preprocess

def test_preprocess_builds_transformer():
    df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0],
        "cat1": ["a", "b", "a"],
        "target": [10, 20, 30]
    })
    pre = preprocess(df, target="target")
    assert hasattr(pre, "transform")
    assert hasattr(pre, "fit")
