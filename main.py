import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error

from src.models import fit_nar, recursive_forecast
from src.utils import save_results
from src.data_processing import load_dataset

# -------------------------------
# 1. Load data (notebook-compatible preprocessing)
# -------------------------------

data = load_dataset("data/data.xlsx")

# -------------------------------
# 1a. Plot SM, P, T time series (replicates notebook fig)
# -------------------------------

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(16, 10), sharex=True)
time_index = data.index

axes[0].plot(time_index, data["SM"], linewidth=1.8)
axes[0].set_ylabel("SM (%)")
axes[0].set_title("Soil Moisture Content (SM), Precipitation (P), and Air Temperature (T)")
axes[0].grid(True, which="major", linestyle="-", alpha=0.25)

axes[1].plot(time_index, data["P"], linewidth=1.6)
axes[1].set_ylabel("P (mm/interval)")
axes[1].grid(True, which="major", linestyle="-", alpha=0.25)

axes[2].plot(time_index, data["T"], linewidth=1.8)
axes[2].set_ylabel("T (°C)")
axes[2].set_xlabel("Time")
axes[2].grid(True, which="major", linestyle="-", alpha=0.25)

locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
formatter = mdates.ConciseDateFormatter(locator)
axes[2].xaxis.set_major_locator(locator)
axes[2].xaxis.set_major_formatter(formatter)

plt.tight_layout()
plt.show()

# -------------------------------
# 2. Train / Test split
# -------------------------------

train = data.loc[:'2019-12-31']
test  = data.loc['2020-01-01':]

print("Train samples:", len(train))
print("Test samples:", len(test))

# Soil Moisture Index (SMI) using train-period statistics
mu_sm = train["SM"].mean()
sd_sm = train["SM"].std(ddof=1)
data["SMI"] = (data["SM"] - mu_sm) / sd_sm
print("mu_sm:", mu_sm)
print("sd_sm:", sd_sm)

# -------------------------------
# 3. Fit NAR models for P and T
# -------------------------------

LAGS_PT = 48

nar_P = fit_nar(train["P"], lags=LAGS_PT)
nar_T = fit_nar(train["T"], lags=LAGS_PT)

# -------------------------------
# 4. Forecast P and T
# -------------------------------

P_hat = recursive_forecast(nar_P, train["P"].values, len(test), LAGS_PT)
T_hat = recursive_forecast(nar_T, train["T"].values, len(test), LAGS_PT)

P_hat = pd.Series(P_hat, index=test.index, name="P_hat")
T_hat = pd.Series(T_hat, index=test.index, name="T_hat")

print("Preview forecasts:")
print(pd.concat([test[["P", "T"]].head(), P_hat.head(), T_hat.head()], axis=1))

# -------------------------------
# 5. SPEI computation (Stage 2)
# -------------------------------

K = 0.05         # temperature-to-demand scaling (proxy)
WINDOW = 720     # ~30 days if sampling is ~hourly

def water_balance(P, T, k=K):
    return P - k * np.maximum(T, 0.0)

# Assemble full P,T series: observed in train, forecasted in test
P_full = pd.concat([train["P"], P_hat])
T_full = pd.concat([train["T"], T_hat])

D_full = pd.Series(
    water_balance(P_full.values, T_full.values),
    index=P_full.index,
    name="D",
)
D_acc = D_full.rolling(WINDOW).sum()

# Standardize using TRAIN statistics only
D_acc_train = D_acc.loc[train.index].dropna()
mu_D = D_acc_train.mean()
sd_D = D_acc_train.std(ddof=1)

SPEI = (D_acc - mu_D) / sd_D
SPEI.name = "SPEI"

print("SPEI train mean/std:", float(mu_D), float(sd_D))

# -------------------------------
# 6. Build NARX features for SMI using SPEI (Stage 3)
# -------------------------------

LAGS_SMI = 48
LAGS_X   = 48

SMI = data["SMI"].copy()
SPEI_all = SPEI.reindex(data.index)

smi_arr = SMI.values.astype(float)
spei_arr = SPEI_all.values.astype(float)

idx = np.arange(len(data.index))
train_end = np.where(data.index <= pd.Timestamp("2019-12-31 23:59:59"))[0][-1]
test_start = np.where(data.index.year == 2020)[0][0]
test_end   = np.where(data.index.year == 2020)[0][-1]

start = max(WINDOW, LAGS_SMI, LAGS_X)
candidates = idx[start:]

mask_target = np.isfinite(smi_arr)
mask_exog   = np.isfinite(spei_arr)
candidates = candidates[mask_target[candidates] & mask_exog[candidates]]

smi_win  = sliding_window_view(smi_arr,  LAGS_SMI)
spei_win = sliding_window_view(spei_arr, LAGS_X)

def build_Xy(t_indices):
    rows_smi = t_indices - LAGS_SMI
    rows_x   = t_indices - LAGS_X
    X = np.hstack([smi_win[rows_smi], spei_win[rows_x]])
    y = smi_arr[t_indices]
    finite = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[finite], y[finite]

train_t = candidates[candidates <= train_end]
test_t  = candidates[(candidates >= test_start) & (candidates <= test_end)]

X_train, y_train = build_Xy(train_t)
X_test,  y_test  = build_Xy(test_t)

print("NARX train shape:", X_train.shape, y_train.shape)
print("NARX test shape :", X_test.shape,  y_test.shape)

narx = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=40,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
    )),
])

narx.fit(X_train, y_train)
pred = narx.predict(X_test)

r2 = r2_score(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print("Test year 2020 — NAR–SPEI–NARX pipeline")
print(f"R^2:  {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# Publication-style plot of observed vs predicted SMI for 2020
plt.style.use("fivethirtyeight")
test_times = data.index[test_t][:len(y_test)]

fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
ax.plot(test_times, y_test, label="Observed SMI", linewidth=2.2)
ax.plot(test_times, pred, label="Predicted SMI", linewidth=2.2, alpha=0.9)

ax.set_title(f"Observed vs Predicted SMI (Year 2020) — $R^2$={r2:.3f}", pad=12)
ax.set_xlabel("Time")
ax.set_ylabel("SMI (z-score)")

try:
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
    )
    fig.autofmt_xdate(rotation=0)
except Exception:
    pass

ax.grid(True, which="major", linewidth=0.8, alpha=0.6)
ax.legend(frameon=True, loc="best")

fig.tight_layout()
plt.show()

# -------------------------------
# 9. Save results
# -------------------------------

results = pd.DataFrame(
    {"SMI_obs": y_test, "SMI_pred": pred},
    index=test_times,
)
save_results(results, "outputs/sm_forecast.csv")