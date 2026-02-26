import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from .data_processing import make_lag_matrix


def fit_nar(series, lags=48, hidden=(64, 32), max_iter=40, random_state=42):
    arr = series.values.astype(float)
    X, y = make_lag_matrix(arr, lags)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5
        ))
    ])

    model.fit(X, y)
    return model


def recursive_forecast(model, history, steps, lags):
    hist = history.astype(float).copy()
    preds = np.empty(steps)

    for i in range(steps):
        x = hist[-lags:].reshape(1, -1)
        preds[i] = model.predict(x)[0]
        hist = np.append(hist, preds[i])

    return preds