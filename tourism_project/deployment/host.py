import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

SPACE_REPO_ID = "nikhileshmehta1989/tourism-package-prediction"
api = HfApi(token=os.getenv("HF_TOKEN"))

# Create the HF Space if it doesn't exist
try:
    api.repo_info(repo_id=SPACE_REPO_ID, repo_type="space")
    print(f"Space '{SPACE_REPO_ID}' already exists.")
except RepositoryNotFoundError:
    create_repo(
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        space_sdk="streamlit",
        private=False,
    )
    print(f"Space '{SPACE_REPO_ID}' created.")

# Upload all deployment files to the HF Space
api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=SPACE_REPO_ID,
    repo_type="space",
)
print(f"All deployment files uploaded to https://huggingface.co/spaces/{SPACE_REPO_ID}")
