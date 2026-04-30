from huggingface_hub import hf_hub_download
import os

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = hf_hub_download(
    repo_id="neuronsbyisshu/brain-tumor-model",
    filename="brain_tumor_final.h5",
    local_dir=MODEL_DIR,
    local_dir_use_symlinks=False
)

print(f"Model downloaded")