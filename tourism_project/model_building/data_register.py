# from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
# from huggingface_hub import HfApi, create_repo
# import os


# repo_id = "nikhileshmehta1989/tourism-package-prediction"
# repo_type = "dataset"

# # Initialize API client
# api = HfApi(token=os.getenv("HF_TOKEN"))

# # Step 1: Check if the space exists
# try:
#     api.repo_info(repo_id=repo_id, repo_type=repo_type)
#     print(f"Space '{repo_id}' already exists. Using it.")
# except RepositoryNotFoundError:
#     print(f"Space '{repo_id}' not found. Creating new space...")
#     create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
#     print(f"Space '{repo_id}' created.")

# api.upload_folder(
#     folder_path="tourism_project/data",
#     repo_id=repo_id,
#     repo_type=repo_type,
# )

from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError
import os

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set")

repo_id = "nikhileshmehta1989/tourism-package-prediction"
repo_type = "dataset"

api = HfApi(token=HF_TOKEN)

print("Authenticated HF user:", api.whoami())

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating it...")
    api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False,
        exist_ok=True,
    )
    print(f"Dataset repo '{repo_id}' created.")

# Upload raw/source dataset folder contents
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)

print("Folder uploaded successfully.")




