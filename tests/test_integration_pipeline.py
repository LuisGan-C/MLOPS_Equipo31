import pandas as pd
from src.train import load_data, clean_data, preprocess, train_pipeline, evaluate_model
from sklearn.linear_model import Ridge

def test_end_to_end_pipeline():
    # Mini dataset simulado
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4],
        "feature2": ["a", "b", "a", "b"],
        "target": [10, 15, 20, 25]
    })
    df_clean = clean_data(df)
    y = df_clean["target"]
    X = df_clean.drop(columns=["target"])
    pre = preprocess(df_clean, "target")
    model = Ridge()
    pipe = train_pipeline(X, y, pre, model)
    mae, rmse, r2 = evaluate_model(pipe, X, y)
    assert 0 <= r2 <= 1
