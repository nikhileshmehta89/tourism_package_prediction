
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set")

api = HfApi(token=HF_TOKEN)

repo_id = "nikhileshmehta1989/tourism-package-prediction"
DATASET_PATH = f"hf://datasets/{repo_id}/tourism.csv"
print("Reading from:", DATASET_PATH)
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

df.drop(columns=["Unnamed: 0", "CustomerID"], inplace=True, errors="ignore")

num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    mode_val = df[col].mode()
    if not mode_val.empty:
        df[col] = df[col].fillna(mode_val.iloc[0])

print("Missing values handled.")

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

print("Categorical columns encoded.")

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

os.makedirs("tourism_project/data", exist_ok=True)
train_path = "tourism_project/data/train.csv"
test_path = "tourism_project/data/test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)
print("Train and test datasets saved locally.")

api.upload_file(
    path_or_fileobj=train_path,
    path_in_repo="train.csv",
    repo_id=repo_id,
    repo_type="dataset",
)

api.upload_file(
    path_or_fileobj=test_path,
    path_in_repo="test.csv",
    repo_id=repo_id,
    repo_type="dataset",
)

print("Train and test datasets uploaded to Hugging Face successfully.")
