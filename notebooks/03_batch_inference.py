# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Batch inference + basic monitoring
# MAGIC
# MAGIC Loads whatever model holds the `@champion` alias (no code change needed when a new version is promoted),
# MAGIC scores the incoming tickets, and writes predictions to a Delta table with the model version for traceability.

# COMMAND ----------

dbutils.widgets.text("catalog", "support_ai")
CATALOG = dbutils.widgets.get("catalog")
MODEL_NAME = f"{CATALOG}.ml.ticket_classifier"

import mlflow
from mlflow import MlflowClient
from pyspark.sql import functions as F

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
print(f"Loaded {MODEL_NAME} v{champion.version}")

# COMMAND ----------

FEATURES = ["text", "channel", "customer_tier"]
incoming = spark.table(f"{CATALOG}.gold.incoming_tickets").toPandas()

out = incoming[["ticket_id", "created_at", "channel", "customer_tier", "text"]].copy()
out["predicted_priority"] = model.predict(incoming[FEATURES])
out["actual_priority"] = incoming["priority"]   # in reality, labels arrive later (feedback loop)
out["model_version"] = int(champion.version)

(spark.createDataFrame(out)
    .withColumn("scored_at", F.current_timestamp())
    .write.mode("append").saveAsTable(f"{CATALOG}.ml.ticket_predictions"))

print(f"Scored {len(out)} tickets")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitoring
# MAGIC 1. **Prediction drift** — does the predicted distribution differ from what the model saw in training?
# MAGIC 2. **Performance** — once true labels arrive, how accurate is the live model?

# COMMAND ----------

display(spark.sql(f"""
WITH train AS (
  SELECT priority AS label, count(*) / sum(count(*)) OVER () AS train_share
  FROM {CATALOG}.gold.ticket_features GROUP BY priority
),
live AS (
  SELECT predicted_priority AS label, count(*) / sum(count(*)) OVER () AS live_share
  FROM {CATALOG}.ml.ticket_predictions
  WHERE model_version = {champion.version}
  GROUP BY predicted_priority
)
SELECT t.label, round(t.train_share, 3) AS train_share, round(l.live_share, 3) AS live_share,
       round(l.live_share - t.train_share, 3) AS diff
FROM train t JOIN live l USING (label)
ORDER BY label
"""))

# COMMAND ----------

display(spark.sql(f"""
SELECT model_version,
       count(*) AS scored,
       round(avg(CASE WHEN predicted_priority = actual_priority THEN 1 ELSE 0 END), 3) AS live_accuracy,
       round(sum(CASE WHEN actual_priority = 'HIGH' AND predicted_priority = 'HIGH' THEN 1 ELSE 0 END)
             / sum(CASE WHEN actual_priority = 'HIGH' THEN 1 ELSE 0 END), 3) AS live_recall_high
FROM {CATALOG}.ml.ticket_predictions
GROUP BY model_version
ORDER BY model_version
"""))
# If live_recall_high drops below an agreed threshold -> alert + trigger retraining (rerun 01 -> 02).
