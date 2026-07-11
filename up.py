import kagglehub

# Ignore specific files and directories (use a trailing slash for directories)
kagglehub.dataset_upload(
    handle="bibocat/viettel-nlp-coes",
    local_dataset_dir="./",
    ignore_patterns=[".venv/", "videos/", ".git/", "logs/", "models/",
                     "output.zip", ".mypy_cache/", "data_train/"],
)
