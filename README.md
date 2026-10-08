# Databricks MLOps Platform

End-to-end MLOps on Databricks: classical ML, **LLM/RAG** and **computer vision**, all tracked in MLflow, versioned in the Unity Catalog model registry, orchestrated as Databricks jobs, and deployed through GitHub Actions with Databricks Asset Bundles.

## What's inside

| Pipeline | Notebooks | What it demonstrates |
|---|---|---|
| Ticket priority (classical ML) | `00` → `01` → `02` → `03` | Medallion Delta pipeline, data quality checks, `MERGE`, time-based split, 3 tracked candidate models, champion/challenger gate, batch inference, drift + live-accuracy monitoring |
| Computer vision | `00` → `10` | Images in a UC Volume + manifest table, PyTorch training loop, per-epoch MLflow metrics, tensor model signature, quality gate, `@champion` batch inference |
| LLM / RAG | `00` → `20` | Docs in a Volume → chunking → Foundation Model embeddings → Delta → cosine retrieval → LLM with citations → evaluation (hit rate, MRR, LLM-as-judge faithfulness, refusal, latency, tokens) → RAG chain packaged as MLflow pyfunc and registered |

## Architecture

```
            ┌─────────── GitHub ───────────┐
 feature ──▶│ PR ─▶ CI: ruff + pytest       │
 branch     │       └▶ bundle validate      │
            │       └▶ bundle deploy -t dev │──▶ Databricks: "[dev user]" jobs
            │ merge ─▶ CD: test             │
            │       └▶ bundle deploy -t prod│──▶ Databricks: prod jobs
            └───────────────────────────────┘
                                                   │
             Unity Catalog: support_ai             ▼
  bronze (raw, Volume) ─▶ silver (clean) ─▶ gold (features / embeddings)
                                                   │
                  MLflow experiments ─▶ UC model registry (@champion / @challenger)
                                                   │
                     batch inference ─▶ ml.ticket_predictions ─▶ monitoring
```

## Repository layout

```
databricks.yml              # Asset Bundle: variables, dev/prod targets
resources/*.job.yml         # Job definitions (DAGs of notebook tasks)
notebooks/                  # Databricks notebooks (source format)
src/rag/                    # chunking, retrieval, prompt (unit tested)
src/cv/                     # synthetic image data, CNN model (unit tested)
tests/                      # pytest, runs in CI without Databricks
.github/workflows/ci.yml    # PR: lint, test, validate, deploy to dev
.github/workflows/cd.yml    # main: test, deploy to prod, optional job run
```

## Setup

1. **Databricks token**: Settings → Developer → Access tokens → Generate.
2. **GitHub secrets** (repo → Settings → Secrets and variables → Actions):
   - `DATABRICKS_HOST` = workspace URL, e.g. `https://dbc-xxxx.cloud.databricks.com`
   - `DATABRICKS_TOKEN` = the token
3. **Interactive development**: Databricks → Workspace → Create → Git folder → this repo. Use **Pull** to sync after pushes.
4. **Local CLI (optional)**: `databricks auth login --host <url>` then `databricks bundle validate` / `deploy`.

If the catalog can't be created (permissions), deploy with `--var="catalog=workspace"`.

## Flow

`feature branch → PR (CI: lint, tests, deploy dev) → merge to main (CD: deploy prod) → Actions → CD → Run workflow → choose a job to run`

## Production upgrades (not used here because of Free Edition / scope)

- Workload identity federation (OIDC) or a service principal instead of a personal token; `run_as` service principal in prod
- Separate catalogs per environment (`support_ai_dev`, `support_ai_prod`)
- AI Search (Vector Search) Delta Sync index in place of brute-force retrieval
- Real-time Model Serving endpoint with inference tables + Lakehouse Monitoring
- Lakeflow Declarative Pipelines with expectations for the data layer
