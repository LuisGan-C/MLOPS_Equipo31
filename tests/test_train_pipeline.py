import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from src.mlops_equipo31.train import train_pipeline, preprocess

def test_train_pipeline_trains_model():
    df = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": ["low", "medium", "high"],
        "target": [10, 15, 20]
    })
    X = df.drop(columns=["target"])
    y = df["target"]
    pre = preprocess(df, target="target")
    model = Ridge()
    pipe = train_pipeline(X, y, pre, model)
    assert hasattr(pipe, "predict")
    preds = pipe.predict(X)
    assert len(preds) == len(y)
