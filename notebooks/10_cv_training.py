# Databricks notebook source
# MAGIC %md
# MAGIC # 10 — Computer vision: PyTorch CNN with MLflow + Unity Catalog
# MAGIC
# MAGIC ```
# MAGIC Images (PNG) → UC Volume → manifest Delta table (bronze.shape_images)
# MAGIC   → load + split (train / val / test) → PyTorch training loop (CPU, serverless)
# MAGIC   → MLflow: per-epoch metrics, confusion matrix, model with tensor signature
# MAGIC   → register ml.shape_classifier → quality gate → @champion → batch inference
# MAGIC ```
# MAGIC The dataset is synthetic (circles / squares / triangles) so it needs no download and trains in about a minute on CPU.
# MAGIC Swapping in a real dataset only changes the "land images" step; the rest of the pipeline is identical.

# COMMAND ----------

# MAGIC %pip install torch --index-url https://download.pytorch.org/whl/cpu
# MAGIC # If that index is blocked, use instead:  %pip install torch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "support_ai")
CATALOG = dbutils.widgets.get("catalog")
IMG_DIR = f"/Volumes/{CATALOG}/bronze/raw_files/shapes"
MODEL_NAME = f"{CATALOG}.ml.shape_classifier"
MIN_ACCURACY = 0.90

PARAMS = {"epochs": 8, "batch_size": 64, "lr": 1e-3, "img_size": 64, "n_per_class": 300, "seed": 42}

import os, sys
sys.path.append(os.path.abspath(".."))
from src.cv.model import SmallCNN
from src.cv.shapes import CLASSES, make_dataset

import numpy as np
import torch

torch.manual_seed(PARAMS["seed"])
device = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device:", device)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Land images in a Volume + manifest table (how unstructured data is governed in Unity Catalog)

# COMMAND ----------

from PIL import Image
import pandas as pd

images, labels = make_dataset(PARAMS["n_per_class"], PARAMS["img_size"], PARAMS["seed"])

manifest = []
for name in CLASSES:
    dbutils.fs.mkdirs(f"{IMG_DIR}/{name}")
for i, (img, label) in enumerate(zip(images, labels)):
    path = f"{IMG_DIR}/{CLASSES[label]}/{i:05d}.png"
    Image.fromarray(img).save(path)
    manifest.append({"path": path, "label": int(label), "label_name": CLASSES[label]})

(spark.createDataFrame(pd.DataFrame(manifest))
    .write.mode("overwrite").option("overwriteSchema", True)
    .saveAsTable(f"{CATALOG}.bronze.shape_images"))
display(spark.sql(f"SELECT label_name, count(*) n FROM {CATALOG}.bronze.shape_images GROUP BY label_name"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load from the Volume and split (stratified train / val / test)

# COMMAND ----------

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

meta = spark.table(f"{CATALOG}.bronze.shape_images").toPandas()
X = np.stack([np.asarray(Image.open(p), dtype=np.float32) / 255.0 for p in meta.path])[:, None, :, :]  # N,1,H,W
y = meta.label.values

X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)
print("train", X_train.shape, "val", X_val.shape, "test", X_test.shape)


def loader(X, y, shuffle):
    return DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long)),
                      batch_size=PARAMS["batch_size"], shuffle=shuffle)


train_dl, val_dl = loader(X_train, y_train, True), loader(X_val, y_val, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Train — the classic PyTorch loop, every epoch logged to MLflow

# COMMAND ----------

import mlflow

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/shape-classifier-cv")


def evaluate(model, dl):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss_sum += loss_fn(out, yb).item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            total += len(yb)
    return loss_sum / total, correct / total


model = SmallCNN(n_classes=len(CLASSES)).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS["lr"])

with mlflow.start_run(run_name="small_cnn") as run:
    mlflow.log_params({**PARAMS, "model": "SmallCNN", "optimizer": "Adam",
                       "n_params": sum(p.numel() for p in model.parameters())})

    for epoch in range(PARAMS["epochs"]):
        model.train()
        running = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()           # 1. reset gradients
            out = model(xb)                 # 2. forward pass
            loss = loss_fn(out, yb)         # 3. compute loss
            loss.backward()                 # 4. backpropagation
            optimizer.step()                # 5. update weights
            running += loss.item() * len(yb)

        val_loss, val_acc = evaluate(model, val_dl)
        mlflow.log_metrics({"train_loss": running / len(X_train), "val_loss": val_loss, "val_accuracy": val_acc},
                           step=epoch)
        print(f"epoch {epoch + 1}: train_loss={running / len(X_train):.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    # ---- Final evaluation on the untouched test set ----
    import matplotlib.pyplot as plt
    from mlflow.models import infer_signature
    from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score

    model.eval()
    with torch.no_grad():
        test_pred = model(torch.tensor(X_test).to(device)).argmax(1).cpu().numpy()
    test_metrics = {"test_accuracy": accuracy_score(y_test, test_pred),
                    "test_f1_macro": f1_score(y_test, test_pred, average="macro")}
    mlflow.log_metrics(test_metrics)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, test_pred, display_labels=CLASSES, ax=ax)
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)

    model.cpu()
    sample = X_test[:4]
    with torch.no_grad():
        sample_out = model(torch.tensor(sample)).numpy()
    info = mlflow.pytorch.log_model(model, "model",
                                    signature=infer_signature(sample, sample_out),
                                    input_example=sample)
    mlflow.log_dict({"classes": CLASSES}, "classes.json")

print(test_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Register + quality gate

# COMMAND ----------

from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

if test_metrics["test_accuracy"] < MIN_ACCURACY:
    raise ValueError(f"Test accuracy {test_metrics['test_accuracy']:.3f} below {MIN_ACCURACY}; not registering")

mv = mlflow.register_model(info.model_uri, MODEL_NAME)
try:
    champ = client.get_model_version_by_alias(MODEL_NAME, "champion")
    champ_acc = client.get_run(champ.run_id).data.metrics.get("test_accuracy", -1)
except Exception:
    champ_acc = -1

alias = "champion" if test_metrics["test_accuracy"] >= champ_acc else "challenger"
client.set_registered_model_alias(MODEL_NAME, alias, mv.version)
print(f"{MODEL_NAME} v{mv.version} -> @{alias} (acc {test_metrics['test_accuracy']:.3f} vs champion {champ_acc:.3f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Batch inference with @champion on brand-new images

# COMMAND ----------

from src.cv.shapes import make_shape_image

rng = np.random.default_rng(2026)
truth = [CLASSES[i % 3] for i in range(9)]
new_imgs = np.stack([make_shape_image(s, PARAMS["img_size"], rng) for s in truth]).astype(np.float32)[:, None] / 255.0

champion_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
pred = [CLASSES[i] for i in np.asarray(champion_model.predict(new_imgs)).argmax(1)]

fig, axes = plt.subplots(1, 9, figsize=(16, 2.4))
for ax, img, t, p in zip(axes, new_imgs, truth, pred):
    ax.imshow(img[0], cmap="gray")
    ax.set_title(f"true {t}\npred {p}", color="green" if t == p else "red", fontsize=9)
    ax.axis("off")
plt.show()
