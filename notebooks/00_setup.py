# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Setup: Unity Catalog structure
# MAGIC Creates the catalog, medallion schemas (bronze / silver / gold) plus an `ml` schema, and a Volume for raw files.
# MAGIC
# MAGIC If `CREATE CATALOG` fails with a permission error, set the widget to `workspace` and rerun.

# COMMAND ----------

dbutils.widgets.text("catalog", "support_ai")
CATALOG = dbutils.widgets.get("catalog")

if CATALOG != "workspace":
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"USE CATALOG {CATALOG}")

for schema, comment in [
    ("bronze", "Raw ingested data, kept as-is"),
    ("silver", "Cleaned and validated data"),
    ("gold", "Feature / ML-ready data"),
    ("ml", "Models, predictions, monitoring"),
]:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema} COMMENT '{comment}'")

spark.sql("CREATE VOLUME IF NOT EXISTS bronze.raw_files COMMENT 'Landing zone for raw files'")

# COMMAND ----------

display(spark.sql(f"SHOW SCHEMAS IN {CATALOG}"))
display(spark.sql(f"SHOW VOLUMES IN {CATALOG}.bronze"))

import mlflow, sklearn, sys
print("Python :", sys.version.split()[0])
print("MLflow :", mlflow.__version__)
print("sklearn:", sklearn.__version__)
