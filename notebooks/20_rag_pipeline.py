# Databricks notebook source
# MAGIC %md
# MAGIC # 20 — LLM / RAG pipeline: support-policy assistant
# MAGIC
# MAGIC ```
# MAGIC Policy docs (Volume) → chunk → Delta (silver.doc_chunks)
# MAGIC        → embed (Foundation Model API) → Delta (gold.doc_embeddings)
# MAGIC Question → embed → cosine top-k → prompt + context → LLM → answer with citations
# MAGIC        → evaluate (retrieval hit rate, MRR, faithfulness, refusal, latency) → MLflow
# MAGIC        → package as MLflow pyfunc → register in Unity Catalog (@champion)
# MAGIC ```
# MAGIC
# MAGIC Retrieval is brute-force cosine similarity over a Delta table, which is fine for a few hundred chunks.
# MAGIC The production upgrade is a **Databricks AI Search (Vector Search) Delta Sync index** on `silver.doc_chunks`
# MAGIC with managed embeddings: same pipeline, the index just replaces `top_k`.

# COMMAND ----------

dbutils.widgets.text("catalog", "support_ai")
dbutils.widgets.text("embedding_endpoint", "databricks-gte-large-en")
dbutils.widgets.text("llm_endpoint", "databricks-meta-llama-3-3-70b-instruct")

CATALOG = dbutils.widgets.get("catalog")
EMBED_ENDPOINT = dbutils.widgets.get("embedding_endpoint")
LLM_ENDPOINT = dbutils.widgets.get("llm_endpoint")
DOCS_DIR = f"/Volumes/{CATALOG}/bronze/raw_files/docs"
MODEL_NAME = f"{CATALOG}.ml.support_rag"

CHUNK_SIZE, CHUNK_OVERLAP, TOP_K = 400, 80, 3
MIN_HIT_RATE = 0.8   # quality gate for registering the RAG chain

import os, sys
sys.path.append(os.path.abspath(".."))   # repo root, so `src` is importable
from src.rag.chunking import chunk_text
from src.rag.retrieval import SYSTEM_PROMPT, build_prompt, top_k

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Check which Foundation Model endpoints exist in your workspace
# MAGIC If the defaults above aren't listed, copy the right names into the widgets.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

w = WorkspaceClient()
names = sorted(e.name for e in w.serving_endpoints.list())
print("Embedding-like:", [n for n in names if any(t in n for t in ("gte", "bge", "embed"))])
print("Chat LLM-like :", [n for n in names if any(t in n for t in ("llama", "gpt", "claude", "gemma", "qwen", "mixtral"))])
assert EMBED_ENDPOINT in names, f"Embedding endpoint '{EMBED_ENDPOINT}' not found — update the widget"
assert LLM_ENDPOINT in names, f"LLM endpoint '{LLM_ENDPOINT}' not found — update the widget"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Land source documents in a Unity Catalog Volume (unstructured data)

# COMMAND ----------

DOCS = {
    "refund_policy": """Refund Policy

Customers can request a refund within 30 days of delivery. Items must be unused and in their original packaging.

Approved refunds are returned to the original payment method within 5 to 7 business days. Bank processing can add up to 3 extra days.

If a refund has not arrived after 10 business days, the customer should contact support with the order reference so the finance team can trace it.

Digital products and gift cards are not refundable. Customers may choose store credit instead of a refund, which is issued instantly.""",

    "payment_policy": """Payment Policy

We accept Visa, Mastercard, eSewa, Khalti and bank transfer. Cash on delivery is available only inside Kathmandu Valley.

If a payment fails, no order is created. If money was deducted but the order failed, the amount is automatically reversed within 48 hours.

After three failed payment attempts within one hour, the account's checkout is locked for 30 minutes to prevent fraud.

Customers charged twice for the same order receive the duplicate amount back within 3 business days after support confirms it.""",

    "shipping_policy": """Shipping Policy

Standard shipping takes 3 to 5 business days inside Kathmandu Valley and 5 to 8 business days elsewhere in Nepal.

Express shipping delivers within 24 hours inside Kathmandu Valley for orders placed before 2 PM.

The delivery address can be changed only before the order is dispatched. After dispatch, contact the courier directly.

If a package is marked delivered but not received, report it within 48 hours. We investigate with the courier and resend or refund within 7 days.""",

    "account_security": """Account Security Policy

Two-factor authentication codes are valid for 5 minutes. If a code does not arrive, wait 60 seconds before requesting a new one.

After five incorrect password attempts the account is locked for 15 minutes. Customers can unlock it immediately through the password reset email.

If an account is compromised, support freezes it, revokes all active sessions and requires identity verification before restoring access.

We never ask for passwords or one-time codes by phone, chat or email.""",
}

dbutils.fs.mkdirs(DOCS_DIR)
for name, text in DOCS.items():
    with open(f"{DOCS_DIR}/{name}.md", "w") as f:
        f.write(text)
display(dbutils.fs.ls(DOCS_DIR))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Chunk → Delta (`silver.doc_chunks`)

# COMMAND ----------

import pandas as pd

rows = []
for fname in sorted(os.listdir(DOCS_DIR)):
    source = fname.rsplit(".", 1)[0]
    with open(f"{DOCS_DIR}/{fname}") as f:
        for i, chunk in enumerate(chunk_text(f.read(), CHUNK_SIZE, CHUNK_OVERLAP)):
            rows.append({"chunk_id": f"{source}_{i:03d}", "source": source, "text": chunk})

chunks_pdf = pd.DataFrame(rows)
(spark.createDataFrame(chunks_pdf)
    .write.mode("overwrite").option("overwriteSchema", True)
    .saveAsTable(f"{CATALOG}.silver.doc_chunks"))
print(f"{len(chunks_pdf)} chunks from {chunks_pdf.source.nunique()} documents")
display(chunks_pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Embed → Delta (`gold.doc_embeddings`)

# COMMAND ----------

def embed(texts, batch_size=16):
    vectors = []
    for i in range(0, len(texts), batch_size):
        resp = w.serving_endpoints.query(name=EMBED_ENDPOINT, input=texts[i:i + batch_size])
        vectors.extend(d.embedding for d in resp.data)
    return vectors


chunks_pdf["embedding"] = embed(chunks_pdf["text"].tolist())
(spark.createDataFrame(chunks_pdf)
    .write.mode("overwrite").option("overwriteSchema", True)
    .saveAsTable(f"{CATALOG}.gold.doc_embeddings"))
print("Embedding dimension:", len(chunks_pdf["embedding"][0]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Retrieve + generate

# COMMAND ----------

import time
import numpy as np

index = spark.table(f"{CATALOG}.gold.doc_embeddings").toPandas()
matrix = np.vstack(index["embedding"].values)


def retrieve(question, k=TOP_K):
    q_vec = embed([question])[0]
    idx, scores = top_k(q_vec, matrix, k)
    return [{"chunk_id": index.chunk_id[i], "source": index.source[i], "text": index.text[i],
             "score": float(s)} for i, s in zip(idx, scores)]


def chat(system, user, max_tokens=300):
    resp = w.serving_endpoints.query(
        name=LLM_ENDPOINT,
        messages=[ChatMessage(role=ChatMessageRole.SYSTEM, content=system),
                  ChatMessage(role=ChatMessageRole.USER, content=user)],
        max_tokens=max_tokens, temperature=0.0,
    )
    tokens = getattr(getattr(resp, "usage", None), "total_tokens", None) or 0
    return resp.choices[0].message.content, tokens


def answer(question):
    start = time.time()
    contexts = retrieve(question)
    text, tokens = chat(SYSTEM_PROMPT, build_prompt(question, contexts))
    return {"answer": text, "contexts": contexts, "latency_s": time.time() - start, "tokens": tokens}


demo = answer("My payment failed but money was taken from my account. What happens now?")
print(demo["answer"])
print("\nRetrieved:", [(c["chunk_id"], round(c["score"], 3)) for c in demo["contexts"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Evaluate and track in MLflow
# MAGIC - **Retrieval**: hit rate@k (was the right document retrieved?) and MRR (how high was it ranked?)
# MAGIC - **Generation**: faithfulness via LLM-as-judge (is the answer supported by the retrieved context?)
# MAGIC - **Safety**: refusal on an out-of-scope question (does it say "I don't know" instead of hallucinating?)
# MAGIC - **Operational**: latency and token usage

# COMMAND ----------

EVAL_SET = [
    ("How many days do I have to request a refund?", "refund_policy"),
    ("How long until my refund reaches my card?", "refund_policy"),
    ("I was charged twice for one order, when do I get my money back?", "payment_policy"),
    ("What happens after three failed payment attempts?", "payment_policy"),
    ("Can I pay cash on delivery in Pokhara?", "payment_policy"),
    ("How long does standard shipping take outside the valley?", "shipping_policy"),
    ("My parcel says delivered but I never got it", "shipping_policy"),
    ("My two factor code is not arriving", "account_security"),
    ("I typed my password wrong too many times and got locked out", "account_security"),
]
OUT_OF_SCOPE = "What is the CEO's favourite football team?"

JUDGE_PROMPT = ("You are grading a support bot. Reply with only YES or NO: "
                "is every claim in the ANSWER supported by the CONTEXT?")

import mlflow

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/rag-support-assistant")

with mlflow.start_run(run_name=f"rag_chunk{CHUNK_SIZE}_k{TOP_K}") as rag_run:
    mlflow.log_params({"embedding_endpoint": EMBED_ENDPOINT, "llm_endpoint": LLM_ENDPOINT,
                       "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP, "top_k": TOP_K,
                       "n_chunks": len(index), "retrieval": "brute_force_cosine"})
    mlflow.log_text(SYSTEM_PROMPT, "system_prompt.txt")

    records = []
    for question, expected in EVAL_SET:
        res = answer(question)
        sources = [c["source"] for c in res["contexts"]]
        rank = sources.index(expected) + 1 if expected in sources else None
        context_text = "\n\n".join(c["text"] for c in res["contexts"])
        verdict, _ = chat(JUDGE_PROMPT, f"CONTEXT:\n{context_text}\n\nANSWER:\n{res['answer']}", max_tokens=5)
        records.append({
            "question": question, "expected_source": expected, "retrieved_sources": ", ".join(sources),
            "hit": rank is not None, "reciprocal_rank": 1 / rank if rank else 0.0,
            "faithful": verdict.strip().upper().startswith("YES"),
            "latency_s": round(res["latency_s"], 2), "tokens": res["tokens"], "answer": res["answer"],
        })

    oos = answer(OUT_OF_SCOPE)
    refused = "don't know" in oos["answer"].lower() or "do not know" in oos["answer"].lower()

    eval_df = pd.DataFrame(records)
    metrics = {
        "hit_rate_at_k": eval_df.hit.mean(),
        "mrr": eval_df.reciprocal_rank.mean(),
        "faithfulness_rate": eval_df.faithful.mean(),
        "refused_out_of_scope": float(refused),
        "avg_latency_s": eval_df.latency_s.mean(),
        "total_tokens": float(eval_df.tokens.sum()),
    }
    mlflow.log_metrics(metrics)
    mlflow.log_table(eval_df, "eval_results.json")

print(metrics)
print("Out-of-scope answer:", oos["answer"])
display(eval_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Package the RAG chain as an MLflow model and register it (versioned like any other model)
# MAGIC The chunk embeddings and config travel with the model as artifacts, so a model version pins
# MAGIC *exactly* which documents, chunking, prompt and endpoints produced its answers.

# COMMAND ----------

import json
from mlflow import MlflowClient
from mlflow.models import infer_signature

ARTIFACT_DIR = "/tmp/rag_artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
index.to_parquet(f"{ARTIFACT_DIR}/index.parquet")
with open(f"{ARTIFACT_DIR}/config.json", "w") as f:
    json.dump({"embedding_endpoint": EMBED_ENDPOINT, "llm_endpoint": LLM_ENDPOINT, "top_k": TOP_K}, f)


class SupportRAG(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import pandas as pd
        from databricks.sdk import WorkspaceClient
        self.index = pd.read_parquet(context.artifacts["index"])
        self.matrix = np.vstack(self.index["embedding"].values)
        with open(context.artifacts["config"]) as f:
            self.cfg = json.load(f)
        self.w = WorkspaceClient()

    def predict(self, context, model_input, params=None):
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
        from src.rag.retrieval import SYSTEM_PROMPT, build_prompt, top_k
        answers = []
        for q in model_input["question"]:
            q_vec = self.w.serving_endpoints.query(name=self.cfg["embedding_endpoint"], input=[q]).data[0].embedding
            idx, _ = top_k(q_vec, self.matrix, self.cfg["top_k"])
            ctx = [{"source": self.index.source[i], "text": self.index.text[i]} for i in idx]
            resp = self.w.serving_endpoints.query(
                name=self.cfg["llm_endpoint"],
                messages=[ChatMessage(role=ChatMessageRole.SYSTEM, content=SYSTEM_PROMPT),
                          ChatMessage(role=ChatMessageRole.USER, content=build_prompt(q, ctx))],
                max_tokens=300, temperature=0.0)
            answers.append(resp.choices[0].message.content)
        return answers


example = pd.DataFrame({"question": ["What is the refund period?"]})
with mlflow.start_run(run_id=rag_run.info.run_id):
    info = mlflow.pyfunc.log_model(
        "rag_model",
        python_model=SupportRAG(),
        artifacts={"index": f"{ARTIFACT_DIR}/index.parquet", "config": f"{ARTIFACT_DIR}/config.json"},
        code_paths=[os.path.abspath("../src")],
        signature=infer_signature(example, ["Customers can request a refund within 30 days."]),
        input_example=example,
        pip_requirements=["mlflow", "databricks-sdk", "numpy", "pandas", "pyarrow"],
    )

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

if metrics["hit_rate_at_k"] < MIN_HIT_RATE or not refused:
    raise ValueError(f"Quality gate failed: hit_rate={metrics['hit_rate_at_k']:.2f}, refused_oos={refused}")

mv = mlflow.register_model(info.model_uri, MODEL_NAME)
client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
print(f"Registered {MODEL_NAME} v{mv.version} as @champion")

# Smoke test: load the registered chain exactly like a consumer would
rag = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
print(rag.predict(pd.DataFrame({"question": ["Can I change my delivery address after dispatch?"]})))
