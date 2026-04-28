from huggingface_hub import hf_hub_download


hf_hub_download(
    repo_id="neuronsbyisshu/brain-tumor-model",
    filename="brain_tumor_final.keras",
    local_dir="model"
)
print("Model downloaded successfully!")