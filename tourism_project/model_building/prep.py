# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi


api = HfApi(token=os.getenv("HF_TOKEN"))

# Define constants for the dataset and output paths
DATASET_PATH = "hf://datasets/nikhileshmehta89/tourism_package_prediction/tourism_package.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# ── Data Cleaning ──────────────────────────────────────────────────────────────
# Drop the unnamed index column and CustomerID (not useful for prediction)
df.drop(columns=["Unnamed: 0", "CustomerID"], inplace=True, errors="ignore")

# Fill missing values: median for numeric, mode for categorical
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()

for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("Missing values handled.")

# Encode categorical columns using LabelEncoder
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print("Categorical columns encoded.")

# ── Train / Test Split ─────────────────────────────────────────────────────────

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_df = X_train.copy()
train_df["ProdTaken"] = y_train.values

test_df = X_test.copy()
test_df["ProdTaken"] = y_test.values

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

# ── Save Locally ───────────────────────────────────────────────────────────────

os.makedirs("tourism_project/data", exist_ok=True)
train_df.to_csv("tourism_project/data/train.csv", index=False)
test_df.to_csv("tourism_project/data/test.csv", index=False)
print("Train and test datasets saved locally.")

# ── Upload to Hugging Face ─────────────────────────────────────────────────────

api.upload_file(
    path_or_fileobj="tourism_project/data/train.csv",
    path_in_repo="train.csv",
    repo_id="nikhileshmehta1989/tourism-package-prediction",
    repo_type="dataset",
)

api.upload_file(
    path_or_fileobj="tourism_project/data/test.csv",
    path_in_repo="test.csv",
    repo_id="nikhileshmehta1989/tourism-package-prediction",
    repo_type="dataset",
)

print("Train and test datasets uploaded to Hugging Face successfully.")
