import numpy as np
import pandas as pd
import os, warnings
warnings.filterwarnings("ignore")

# 🔧 Fix multiprocessing issue
import multiprocessing
multiprocessing.set_start_method("fork", force=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, matthews_corrcoef
)
from sklearn.cluster import KMeans

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# ============================================================
# CONFIG
# ============================================================
BASE_PATH = "/Volumes/SamsungT7/Desktop/Datasets"

DATASETS = {
    "CICIDS": "CICIDS_MASTER.csv",
    "UNSW": "UNSW_MASTER.csv"
}

OUT_BASE = "/Volumes/SamsungT7/Desktop/FINAL_RESULTS_PCA"
os.makedirs(OUT_BASE, exist_ok=True)

TEST_SIZE = 0.6
RANDOM_STATE = 42

FP_COST = 5
FN_COST = 10

# ============================================================
# LOAD DATA
# ============================================================
def load_dataset(path, name):

    print(f"\n📦 Loading {name}...")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        raise ValueError(f"{name} → Label missing")

    df["Label"] = df["Label"].astype(int)

    y = df["Label"].copy()
    df = df.drop(columns=["Label"])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    df = df.select_dtypes(include=[np.number])
    df["Label"] = y.loc[df.index]

    print("Shape:", df.shape)
    print(df["Label"].value_counts())

    return df

# ============================================================
# PIPELINE
# ============================================================
def run_pipeline(df, dataset_name):

    print(f"\n🚀 Running pipeline for {dataset_name}")

    OUT_DIR = os.path.join(OUT_BASE, dataset_name)
    os.makedirs(OUT_DIR, exist_ok=True)

    X = df.drop(columns=["Label"]).values.astype(np.float32)
    y = df["Label"].values

    # SPLIT
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE,
        stratify=y, random_state=RANDOM_STATE
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5,
        stratify=y_temp, random_state=RANDOM_STATE
    )

    # PREPROCESS
    imputer = SimpleImputer()
    X_train = imputer.fit_transform(X_train)
    X_val   = imputer.transform(X_val)
    X_test  = imputer.transform(X_test)

    vt = VarianceThreshold(0.0005)
    X_train = vt.fit_transform(X_train)
    X_val   = vt.transform(X_val)
    X_test  = vt.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # ============================================================
    # PCA
    # ============================================================
    print("\n📉 Applying PCA...")

    pca = PCA(n_components=0.95, svd_solver='full', random_state=42)

    X_train = pca.fit_transform(X_train)
    X_val   = pca.transform(X_val)
    X_test  = pca.transform(X_test)

    explained_variance = np.sum(pca.explained_variance_ratio_)

    print("PCA Components:", X_train.shape[1])
    print("Explained Variance:", explained_variance)

    with open(f"{OUT_DIR}/explained_variance.txt", "w") as f:
        f.write(str(explained_variance))

    # ============================================================
    # MODELS
    # ============================================================
    models = {

        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=16,
            min_samples_split=5,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1
        ),

        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=150,
            max_depth=14,
            class_weight="balanced",
            n_jobs=-1
        ),

        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=80,
            learning_rate=0.05
        ),

        "AdaBoost": AdaBoostClassifier(n_estimators=80),

        "DecisionTree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=15,
            class_weight="balanced"
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            algorithm='kd_tree',
            n_jobs=-1
        ),

        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            solver='saga',
            penalty='l2',
            C=0.5,
            class_weight='balanced',
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            eval_metric='logloss',
            n_jobs=-1
        )
    }

    results = []

    best_score_global = -1
    best_probs = None
    best_threshold = None

    thresholds = np.linspace(0.3, 0.7, 15)

    # ============================================================
    # MODEL LOOP
    # ============================================================
    for name, model in models.items():

        print(f"\n🔹 Training: {name}")

        model.fit(X_train, y_train)

        probs_val = model.predict_proba(X_val)[:,1]

        best_t, best_score = 0.5, -1

        # 🔥 FIXED COST-SENSITIVE THRESHOLD
        for t in thresholds:

            preds = (probs_val >= t).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()

            mcc = matthews_corrcoef(y_val, preds)

            cost = FN_COST * fn + FP_COST * fp
            norm_cost = cost / len(y_val)

            score = 0.8*mcc - 0.2*norm_cost

            if score > best_score:
                best_score = score
                best_t = t

        probs = model.predict_proba(X_test)[:,1]
        preds = (probs >= best_t).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        cost = FP_COST*fp + FN_COST*fn

        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        mcc = matthews_corrcoef(y_test, preds)

        final_score = 0.7*mcc + 0.2*auc + 0.1*f1

        results.append([
            name, acc, f1, auc, mcc, cost, final_score
        ])

        if final_score > best_score_global:
            best_score_global = final_score
            best_probs = probs
            best_threshold = best_t

    # ============================================================
    # RESULTS
    # ============================================================
    df_res = pd.DataFrame(results, columns=[
        "model","accuracy","f1","auc","mcc","cost","score"
    ])

    best = df_res.sort_values("score", ascending=False).iloc[0]

    print("\n🏆 BEST MODEL:")
    print(best)

    df_res.to_csv(f"{OUT_DIR}/results.csv", index=False)
    pd.DataFrame([best]).to_csv(f"{OUT_DIR}/best_model.csv", index=False)

    with open(f"{OUT_DIR}/threshold.txt", "w") as f:
        f.write(str(best_threshold))

    # ============================================================
    # DSI
    # ============================================================
    print("\n📊 Calculating DSI...")

    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_test)

    cluster_mcc = []

    for cid in np.unique(clusters):

        idx = clusters == cid
        if idx.sum() < 30:
            continue

        y_c = y_test[idx]
        p_c = best_probs[idx]
        pred_c = (p_c >= best_threshold).astype(int)

        if len(np.unique(y_c)) < 2:
            continue

        cluster_mcc.append(matthews_corrcoef(y_c, pred_c))

    dsi = 1 - np.std(cluster_mcc) if len(cluster_mcc) > 1 else 0

    print("DSI:", dsi)

    with open(f"{OUT_DIR}/dsi.txt", "w") as f:
        f.write(str(dsi))


# ============================================================
# RUN
# ============================================================
for name, file in DATASETS.items():

    df = load_dataset(os.path.join(BASE_PATH, file), name)
    run_pipeline(df, name)

print("\n🎯 ALL DATASETS COMPLETED")