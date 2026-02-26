import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from .data_processing import make_narx_matrix


def fit_narx(y_series, X_exog_df, lags_y=12, lags_x=12,
             hidden=(64, 32), max_iter=500, random_state=42):

    y = y_series.values.astype(float)
    X_exog = X_exog_df.values.astype(float)

    X, target = make_narx_matrix(y, X_exog, lags_y, lags_x)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True
        ))
    ])

    model.fit(X, target)
    return model


def recursive_forecast_narx(model, y_history, X_future,
                            steps, lags_y, lags_x=12):

    y_hist = y_history.astype(float).copy()
    preds = np.empty(steps)

    for i in range(steps):
        y_lags = y_hist[-lags_y:]
        x_lags = X_future[i - lags_x:i].flatten() if i >= lags_x else \
                 X_future[:lags_x].flatten()

        X_input = np.concatenate([y_lags, x_lags]).reshape(1, -1)

        preds[i] = model.predict(X_input)[0]
        y_hist = np.append(y_hist, preds[i])

    return preds