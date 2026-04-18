import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ── Load Train / Test Data from Hugging Face ───────────────────────────────────

TRAIN_PATH = "hf://datasets/nikhileshmehta89/tourism_package_prediction/train.csv"
TEST_PATH  = "hf://datasets/nikhileshmehta89/tourism_package_prediction/test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

X_train = train_df.drop(columns=["ProdTaken"])
y_train = train_df["ProdTaken"]

X_test  = test_df.drop(columns=["ProdTaken"])
y_test  = test_df["ProdTaken"]

# ── MLflow Experiment Setup ────────────────────────────────────────────────────

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Tourism_Package_Prediction")

# ── Define Models and Parameter Grids ─────────────────────────────────────────

models = {
    "DecisionTree": (
        DecisionTreeClassifier(random_state=42),
        {
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy"],
        },
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
        },
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "n_estimators": [50, 100],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [3, 5],
        },
    ),
}

# ── Train, Tune, and Log Each Model ───────────────────────────────────────────

best_model = None
best_model_name = ""
best_f1 = 0.0

for model_name, (estimator, param_grid) in models.items():
    print(f"\nTuning {model_name}...")

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    best_estimator = grid_search.best_estimator_
    best_params    = grid_search.best_params_

    y_pred = best_estimator.predict(X_test)
    y_prob = best_estimator.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "f1_score":  f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_prob),
    }

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_estimator, artifact_path="model")

    print(f"  Best params : {best_params}")
    print(f"  F1 Score    : {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC     : {metrics['roc_auc']:.4f}")

    if metrics["f1_score"] > best_f1:
        best_f1 = metrics["f1_score"]
        best_model = best_estimator
        best_model_name = model_name

print(f"\nBest model: {best_model_name} with F1 = {best_f1:.4f}")

# ── Save Best Model Locally ────────────────────────────────────────────────────

os.makedirs("tourism_project/model_building", exist_ok=True)
model_path = "tourism_project/model_building/best_model.pkl"
joblib.dump(best_model, model_path)
print(f"Best model saved to {model_path}")

# ── Register Best Model to Hugging Face Model Hub ─────────────────────────────

MODEL_REPO_ID = "nikhileshmehta89/tourism-best-model"
api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=MODEL_REPO_ID, repo_type="model")
    print(f"Model repo '{MODEL_REPO_ID}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=MODEL_REPO_ID, repo_type="model", private=False)
    print(f"Model repo '{MODEL_REPO_ID}' created.")

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="best_model.pkl",
    repo_id=MODEL_REPO_ID,
    repo_type="model",
)
print(f"Best model ({best_model_name}) uploaded to Hugging Face model hub.")
