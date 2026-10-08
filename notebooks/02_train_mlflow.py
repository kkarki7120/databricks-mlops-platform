# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Train, track with MLflow, register in Unity Catalog
# MAGIC
# MAGIC 1. Load Gold features
# MAGIC 2. Train 3 candidate models, each as an MLflow **run** (params, metrics, confusion matrix, model, dataset)
# MAGIC 3. Pick the best by macro F1
# MAGIC 4. Register it in the Unity Catalog model registry
# MAGIC 5. **Validation gate**: only move the `@champion` alias if it beats the current champion

# COMMAND ----------

dbutils.widgets.text("catalog", "support_ai")
CATALOG = dbutils.widgets.get("catalog")
MODEL_NAME = f"{CATALOG}.ml.ticket_classifier"
MIN_F1 = 0.70  # minimum quality bar for any model to be promoted

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")   # registry lives in Unity Catalog
user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/ticket-priority-classifier")

# COMMAND ----------

from sklearn.model_selection import train_test_split

df = spark.table(f"{CATALOG}.gold.ticket_features").toPandas()
FEATURES = ["text", "channel", "customer_tier"]
TARGET = "priority"

X_train, X_test, y_train, y_test = train_test_split(
    df[FEATURES], df[TARGET], test_size=0.2, stratify=df[TARGET], random_state=42
)
print("Train:", X_train.shape, "Test:", X_test.shape)
print(y_train.value_counts(normalize=True).round(3))

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_pipeline(clf, max_features):
    # Preprocessing lives INSIDE the model pipeline, so training and serving
    # always apply identical transformations (no training/serving skew).
    pre = ColumnTransformer([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=max_features), "text"),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["channel", "customer_tier"]),
    ])
    return Pipeline([("pre", pre), ("clf", clf)])


CANDIDATES = {
    "logreg_C1": (LogisticRegression(C=1.0, max_iter=1000), 5000),
    "logreg_C10": (LogisticRegression(C=10.0, max_iter=1000), 5000),
    "random_forest": (RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1), 3000),
}

results = []
for run_name, (clf, max_features) in CANDIDATES.items():
    with mlflow.start_run(run_name=run_name) as run:
        pipe = build_pipeline(clf, max_features)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        params = {"model_type": type(clf).__name__, "tfidf_max_features": max_features,
                  "train_rows": len(X_train)}
        params.update({k: v for k, v in clf.get_params().items() if k in ("C", "n_estimators", "max_depth")})
        mlflow.log_params(params)

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1_macro": f1_score(y_test, preds, average="macro"),
            "precision_macro": precision_score(y_test, preds, average="macro"),
            "recall_macro": recall_score(y_test, preds, average="macro"),
            # Missing a HIGH-priority ticket is the costly mistake, so track it separately
            "recall_high": recall_score(y_test, preds, labels=["HIGH"], average="macro"),
        }
        mlflow.log_metrics(metrics)

        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        try:  # dataset lineage
            mlflow.log_input(mlflow.data.from_pandas(df, name=f"{CATALOG}.gold.ticket_features"), context="training")
        except Exception as e:
            print("Dataset logging skipped:", e)

        signature = infer_signature(X_train.head(5), pipe.predict(X_train.head(5)))
        info = mlflow.sklearn.log_model(pipe, "model", signature=signature, input_example=X_train.head(5))

        results.append({"run_name": run_name, "run_id": run.info.run_id, "model_uri": info.model_uri, **metrics})
        print(f"{run_name:15s} f1_macro={metrics['f1_macro']:.3f} recall_high={metrics['recall_high']:.3f}")

# COMMAND ----------

import pandas as pd

results_df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
display(results_df.drop(columns=["model_uri"]))
best = results_df.iloc[0].to_dict()
print("Best:", best["run_name"], round(best["f1_macro"], 3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register + validation gate (champion / challenger)

# COMMAND ----------

client = MlflowClient()

if best["f1_macro"] < MIN_F1:
    raise ValueError(f"Best F1 {best['f1_macro']:.3f} below minimum {MIN_F1}. Not registering.")

mv = mlflow.register_model(best["model_uri"], MODEL_NAME)
print(f"Registered {MODEL_NAME} version {mv.version}")

try:
    champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    champion_f1 = client.get_run(champion.run_id).data.metrics.get("f1_macro", -1)
except Exception:
    champion, champion_f1 = None, -1  # first ever model

if best["f1_macro"] >= champion_f1:
    client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
    print(f"Version {mv.version} promoted to @champion (f1 {best['f1_macro']:.3f} vs previous {champion_f1:.3f})")
else:
    client.set_registered_model_alias(MODEL_NAME, "challenger", mv.version)
    print(f"Version {mv.version} set as @challenger; champion v{champion.version} kept")

client.update_model_version(MODEL_NAME, mv.version,
                            description=f"{best['run_name']} | f1_macro={best['f1_macro']:.3f}")
