# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Data pipeline: raw CSV → Bronze → Silver → Gold
# MAGIC
# MAGIC | Layer | Table | Purpose |
# MAGIC |---|---|---|
# MAGIC | Raw | Volume `bronze.raw_files/tickets` | Landing zone (CSV files) |
# MAGIC | Bronze | `bronze.tickets_raw` | Exact copy of raw data + ingestion metadata, all strings |
# MAGIC | Silver | `silver.tickets` | Cleaned, typed, deduplicated, validated — upserted with `MERGE` |
# MAGIC | Gold | `gold.ticket_features` / `gold.incoming_tickets` | ML-ready features, split by time |
# MAGIC
# MAGIC The raw data is deliberately dirty (nulls, duplicates, bad labels, messy casing) so the Silver layer has real work to do.

# COMMAND ----------

dbutils.widgets.text("catalog", "support_ai")
CATALOG = dbutils.widgets.get("catalog")
RAW_DIR = f"/Volumes/{CATALOG}/bronze/raw_files/tickets"
spark.sql(f"USE CATALOG {CATALOG}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate synthetic support tickets (stand-in for a source system export)

# COMMAND ----------

import random
from datetime import datetime, timedelta
import pandas as pd

random.seed(42)

# (priority, phrase) per category. Priority is mostly driven by the text,
# with some noise and a customer-tier effect so the model has something to learn.
PHRASES = {
    "PAYMENT": [
        ("HIGH", "payment failed but money was deducted"),
        ("HIGH", "i was charged twice for the same order"),
        ("MEDIUM", "my card was declined at checkout"),
        ("LOW", "which payment methods do you accept"),
    ],
    "ACCOUNT": [
        ("HIGH", "my account was hacked and the password changed"),
        ("HIGH", "cannot log in, locked out of my account"),
        ("MEDIUM", "two factor code is not arriving"),
        ("LOW", "how do i change my profile picture"),
    ],
    "TECHNICAL": [
        ("HIGH", "the app crashes every time and the service is down"),
        ("MEDIUM", "pages are loading very slowly today"),
        ("MEDIUM", "notifications stopped working"),
        ("LOW", "is there a dark mode option"),
    ],
    "SHIPPING": [
        ("HIGH", "package marked delivered but never received"),
        ("MEDIUM", "my order is delayed by a week"),
        ("LOW", "can i change my delivery address"),
        ("LOW", "how long does standard shipping take"),
    ],
    "REFUND": [
        ("HIGH", "refund not received after 30 days"),
        ("MEDIUM", "what is the status of my refund request"),
        ("LOW", "what is your refund policy"),
        ("LOW", "can i get store credit instead of a refund"),
    ],
}
OPENERS = ["Hi,", "Hello team,", "", "Please help,", "Good morning,"]
CLOSERS = ["Thanks.", "Please look into this.", "", "Order ref {o}.", "Regards"]
CHANNELS = ["email", "chat", "phone", "web"]
TIERS = ["free", "pro", "enterprise"]
PRIORITIES = ["HIGH", "MEDIUM", "LOW"]

start = datetime(2026, 4, 1)
rows = []
for i in range(5000):
    category = random.choice(list(PHRASES))
    priority, phrase = random.choice(PHRASES[category])
    tier = random.choices(TIERS, weights=[0.5, 0.35, 0.15])[0]

    # Business rule: enterprise customers often get escalated
    if tier == "enterprise" and priority == "MEDIUM" and random.random() < 0.4:
        priority = "HIGH"
    # Label noise: real-world labels are never perfect
    if random.random() < 0.07:
        priority = random.choice(PRIORITIES)

    order = f"ORD{random.randint(10000, 99999)}"
    description = " ".join(
        p for p in [random.choice(OPENERS), phrase, random.choice(CLOSERS).format(o=order)] if p
    )
    rows.append({
        "ticket_id": f"T{100000 + i}",
        "customer_id": f"C{random.randint(1000, 4999)}",
        "created_at": (start + timedelta(minutes=random.randint(0, 180 * 24 * 60))).strftime("%Y-%m-%d %H:%M:%S"),
        "channel": random.choice(CHANNELS),
        "customer_tier": tier,
        "subject": f"{category.title()} issue",
        "description": description,
        "category": category,
        "priority": priority,
    })

df = pd.DataFrame(rows)

# --- Make it dirty on purpose ---
idx = df.sample(frac=0.01, random_state=1).index
df.loc[idx, "description"] = None                               # missing text
idx = df.sample(frac=0.05, random_state=2).index
df.loc[idx, "priority"] = df.loc[idx, "priority"].str.lower() + " "  # messy casing/whitespace
idx = df.sample(frac=0.005, random_state=3).index
df.loc[idx, "priority"] = "URGENT??"                              # invalid label
df = pd.concat([df, df.sample(frac=0.02, random_state=4)])        # duplicate rows

dbutils.fs.mkdirs(RAW_DIR)
df.to_csv(f"{RAW_DIR}/tickets_batch_001.csv", index=False)
print(f"Wrote {len(df)} raw rows to {RAW_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Bronze — ingest raw files unchanged, add lineage metadata

# COMMAND ----------

from pyspark.sql import functions as F

bronze_df = (
    spark.read.option("header", True).option("escape", '"')
    .csv(RAW_DIR)                                        # everything stays a string in Bronze
    .withColumn("_ingested_at", F.current_timestamp())
    .withColumn("_source_file", F.col("_metadata.file_path"))
)
bronze_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.bronze.tickets_raw")
print("Bronze rows:", spark.table(f"{CATALOG}.bronze.tickets_raw").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Silver — clean, type, deduplicate, validate, then MERGE (upsert)

# COMMAND ----------

raw = spark.table(f"{CATALOG}.bronze.tickets_raw")

clean = (
    raw
    .withColumn("description", F.trim("description"))
    .withColumn("priority", F.upper(F.trim("priority")))
    .withColumn("category", F.upper(F.trim("category")))
    .withColumn("channel", F.lower(F.trim("channel")))
    .withColumn("customer_tier", F.lower(F.trim("customer_tier")))
    .withColumn("created_at", F.to_timestamp("created_at"))
    .filter(F.col("description").isNotNull() & (F.length("description") > 0))
    .filter(F.col("priority").isin("HIGH", "MEDIUM", "LOW"))
    .dropDuplicates(["ticket_id"])
    .select("ticket_id", "customer_id", "created_at", "channel", "customer_tier",
            "subject", "description", "category", "priority")
)

# --- Data quality checks: fail loudly rather than train on bad data ---
raw_count, clean_count = raw.count(), clean.count()
assert clean.filter(F.col("ticket_id").isNull() | F.col("created_at").isNull()).count() == 0, "Null keys/timestamps"
assert clean.select("ticket_id").distinct().count() == clean_count, "Duplicate ticket_ids"
print(f"Raw: {raw_count} | Clean: {clean_count} | Rejected/deduped: {raw_count - clean_count}")

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.silver.tickets (
  ticket_id STRING, customer_id STRING, created_at TIMESTAMP, channel STRING,
  customer_tier STRING, subject STRING, description STRING, category STRING,
  priority STRING, _updated_at TIMESTAMP
)
COMMENT 'Cleaned support tickets'
""")

clean.withColumn("_updated_at", F.current_timestamp()).createOrReplaceTempView("silver_updates")

# Idempotent upsert: re-running the pipeline never creates duplicates
spark.sql(f"""
MERGE INTO {CATALOG}.silver.tickets AS t
USING silver_updates AS s
ON t.ticket_id = s.ticket_id
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *
""")

display(spark.sql(f"SELECT priority, count(*) AS n FROM {CATALOG}.silver.tickets GROUP BY priority ORDER BY n DESC"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Gold — ML-ready features with a **time-based split**
# MAGIC Older tickets → training data (`gold.ticket_features`). The most recent 14 days → `gold.incoming_tickets`,
# MAGIC which simulates *new* tickets arriving after the model was trained. This avoids scoring data the model has already seen.

# COMMAND ----------

silver = spark.table(f"{CATALOG}.silver.tickets")
features = (
    silver
    .withColumn("text", F.lower(F.concat_ws(" ", "subject", "description")))
    .select("ticket_id", "created_at", "channel", "customer_tier", "text", "category", "priority")
)

cutoff = silver.agg(F.max("created_at")).first()[0] - timedelta(days=14)

(features.filter(F.col("created_at") < F.lit(cutoff))
    .write.mode("overwrite").option("overwriteSchema", True)
    .saveAsTable(f"{CATALOG}.gold.ticket_features"))

(features.filter(F.col("created_at") >= F.lit(cutoff))
    .write.mode("overwrite").option("overwriteSchema", True)
    .saveAsTable(f"{CATALOG}.gold.incoming_tickets"))

print("Cutoff:", cutoff)
print("Training rows:", spark.table(f"{CATALOG}.gold.ticket_features").count())
print("Incoming rows:", spark.table(f"{CATALOG}.gold.incoming_tickets").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Delta Lake features worth showing in an interview

# COMMAND ----------

# Every write is a versioned commit in the Delta transaction log
display(spark.sql(f"DESCRIBE HISTORY {CATALOG}.silver.tickets"))

# COMMAND ----------

# Time travel: read the table as it was at version 0
display(spark.sql(f"SELECT count(*) AS rows_at_v0 FROM {CATALOG}.silver.tickets VERSION AS OF 0"))
