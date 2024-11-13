import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    Ridge,
    ElasticNet,
    LogisticRegression,
    lasso_path,
    enet_path,
)

batch_path = lambda n: os.path.join("data", f"batch{n}.pkl")

batch1 = pickle.load(open(batch_path(1), "rb"))
# remove batteries that do not reach 80% capacity
del batch1["b1c8"]
del batch1["b1c10"]
del batch1["b1c12"]
del batch1["b1c13"]
del batch1["b1c22"]

batch2 = pickle.load(open(batch_path(2), "rb"))
batch2_keys = ["b2c7", "b2c8", "b2c9", "b2c15", "b2c16"]
batch1_keys = ["b1c0", "b1c1", "b1c2", "b1c3", "b1c4"]
add_len = [662, 981, 1060, 208, 482]
for i, bk in enumerate(batch1_keys):
    batch1[bk]["cycle_life"] = batch1[bk]["cycle_life"] + add_len[i]
    for j in batch1[bk]["summary"].keys():
        if j == "cycle":
            batch1[bk]["summary"][j] = np.hstack(
                (
                    batch1[bk]["summary"][j],
                    batch2[batch2_keys[i]]["summary"][j]
                    + len(batch1[bk]["summary"][j]),
                )
            )
        else:
            batch1[bk]["summary"][j] = np.hstack(
                (batch1[bk]["summary"][j], batch2[batch2_keys[i]]["summary"][j])
            )
    last_cycle = len(batch1[bk]["cycles"].keys())
    for j, jk in enumerate(batch2[batch2_keys[i]]["cycles"].keys()):
        batch1[bk]["cycles"][str(last_cycle + j)] = batch2[batch2_keys[i]]["cycles"][jk]
del batch2["b2c7"]
del batch2["b2c8"]
del batch2["b2c9"]
del batch2["b2c15"]
del batch2["b2c16"]

batch3 = pickle.load(open(batch_path(3), "rb"))
# remove noisy channels from batch3
del batch3["b3c37"]
del batch3["b3c2"]
del batch3["b3c23"]
del batch3["b3c32"]
del batch3["b3c42"]
del batch3["b3c43"]


features = ["I", "Qc", "Qd", "T", "V"]

FIRST_N_CYCLES = 70
print(f"Using first {FIRST_N_CYCLES} cycles")


def create_summary_df(batch, first_n_cycles):
    N = first_n_cycles
    CYCLE_THRESHOLD = 550

    def classify_battery(num_cycles: int):
        if num_cycles > CYCLE_THRESHOLD:
            return True
        return False

    df_dict = {
        "id": [],
        **{i: [] for i in features},
        "num_cycles": [],
    }

    for battery in batch.keys():
        df_dict["id"].append(battery)
        num_cycles = len(batch[battery]["summary"]["cycle"])
        df_dict["num_cycles"].append(num_cycles)
        for feature in features:
            # print(f"feature={feature} for battery={battery}")
            aucs = np.zeros(N)
            for cycle in [str(i) for i in range(N)]:
                x = batch[battery]["cycles"][str(cycle)]["t"]
                y = batch[battery]["cycles"][str(cycle)][feature]
                auc = np.trapz(y, x)
                aucs[int(cycle)] = auc
            df_dict[feature].append(np.mean(aucs))

    df = pd.DataFrame(df_dict)
    # df["log_nc"] = np.log2(df["num_cycles"])
    # df["num_cycles"] = df["num_cycles"].apply(classify_battery)
    # return df.drop(columns=["num_cycles"])
    return df


def create_principal_components(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features + ["num_cycles"]])
    pca = PCA(n_components=3)
    pca_components = pca.fit_transform(scaled_data)

    explained_variance = pca.explained_variance_ratio_

    # Create a DataFrame for the principal components
    pca_df = pd.DataFrame(pca_components, columns=["PC1", "PC2", "PC3"])

    return pca_df, explained_variance


# make a train set out of first two batches
train_df = create_summary_df({**batch1, **batch2}, FIRST_N_CYCLES)
test_df = create_summary_df(batch3, FIRST_N_CYCLES)

train_df_copy = train_df[train_df["id"] != "b1c18"].copy()
train_df_pca, _ = create_principal_components(train_df_copy)


# test set
test_df_pca, _ = create_principal_components(test_df)
X_test = test_df_pca[[i for i in test_df_pca.columns if i not in ["num_cycles", "id"]]]
Y_test = test_df["num_cycles"]

X_train = train_df_pca[
    [i for i in train_df_pca.columns if i not in ["num_cycles", "id"]]
]
Y_train = train_df_copy["num_cycles"]


rf_model = RandomForestRegressor(
    n_estimators=10_000, random_state=42, max_features="sqrt"
)


rf_model.fit(X_train, Y_train)

y_hat = rf_model.predict(X_train)

# test set
X_test = test_df_pca[[i for i in test_df_pca.columns if i not in ["num_cycles", "id"]]]
Y_test = test_df["num_cycles"]

y_hat = rf_model.predict(X_test)
print(f"test RMSE: {root_mean_squared_error(Y_test, y_hat)}")
