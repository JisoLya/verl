"""Download GSM8K + model, and preprocess into verl-compatible parquet."""
import json
import os
import re

from datasets import load_dataset

DATA_DIR = os.path.expanduser("~/data/gsm8k")
os.makedirs(DATA_DIR, exist_ok=True)


def extract_solution(answer_str: str) -> str:
    m = re.search(r"#### (\-?[0-9\.\,]+)", answer_str)
    assert m is not None, f"No #### in: {answer_str[:100]}"
    return m.group(1).replace(",", "")


INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'


def preprocess(split: str):
    ds = load_dataset("openai/gsm8k", "main", split=split)

    def transform(ex, idx):
        question = ex["question"] + " " + INSTRUCTION
        answer_raw = ex["answer"]
        solution = extract_solution(answer_raw)
        return {
            "data_source": "openai/gsm8k",
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer_raw,
                "question": ex["question"],
            },
        }

    ds = ds.map(transform, with_indices=True)
    out_path = os.path.join(DATA_DIR, f"{split}.parquet")
    ds.to_parquet(out_path)
    print(f"  {split}: {len(ds)} rows -> {out_path}")


print("Downloading & preprocessing GSM8K...")
preprocess("train")
preprocess("test")
print("Data done.")

# Quick verification
import pandas as pd

for s in ("train", "test"):
    df = pd.read_parquet(os.path.join(DATA_DIR, f"{s}.parquet"))
    print(f"  Verify {s}: {len(df)} rows, columns={list(df.columns)}")
    assert len(df) > 0, f"{s} is empty!"
    assert "prompt" in df.columns, "missing prompt column"
    assert "reward_model" in df.columns, "missing reward_model column"

print("Downloading model...")
from modelscope import snapshot_download

snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", cache_dir="/root/.cache/modelscope")
print("Model done.")
