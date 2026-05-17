from datasets import load_dataset
import os

os.makedirs("/root/data/gsm8k", exist_ok=True)
ds = load_dataset("openai/gsm8k", "main")
ds["train"].to_parquet("/root/data/gsm8k/train.parquet")
ds["test"].to_parquet("/root/data/gsm8k/test.parquet")
print("Data done")

from modelscope import snapshot_download

snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", cache_dir="/root/.cache/modelscope")
print("Model done")
