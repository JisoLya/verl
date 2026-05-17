"""demo.py — 1-step GRPO training via API service"""
import requests
import time

SERVICE_URL = "http://localhost:8080"

config = {
    "algorithm": {"adv_estimator": "grpo"},
    "data": {
        "train_files": ["/root/data/gsm8k/train.parquet"],
        "val_files": ["/root/data/gsm8k/test.parquet"],
        "train_batch_size": 16,
        "max_prompt_length": 512,
        "max_response_length": 128,
        "filter_overlong_prompts": True,
        "truncation": "error",
    },
    "actor_rollout_ref": {
        "model": {
            "path": "/root/.cache/modelscope/Qwen/Qwen2.5-0.5B-Instruct",
            "use_remove_padding": False,
            "enable_gradient_checkpointing": True,
        },
        "actor": {
            "optim": {"lr": 5e-7},
            "ppo_mini_batch_size": 8,
            "ppo_micro_batch_size_per_gpu": 1,
            "use_kl_loss": True,
            "kl_loss_coef": 0.001,
            "kl_loss_type": "low_var_kl",
            "fsdp_config": {"param_offload": False, "optimizer_offload": False},
            "use_torch_compile": False,
        },
        "ref": {
            "log_prob_micro_batch_size_per_gpu": 1,
            "fsdp_config": {"param_offload": True},
            "use_torch_compile": False,
        },
        "rollout": {
            "name": "vllm",
            "log_prob_micro_batch_size_per_gpu": 1,
            "tensor_model_parallel_size": 2,
            "gpu_memory_utilization": 0.6,
            "n": 2,
            "enable_chunked_prefill": False,
        },
    },
    "trainer": {
        "critic_warmup": 0,
        "logger": ["console"],
        "project_name": "demo",
        "experiment_name": "api-test",
        "n_gpus_per_node": 8,
        "nnodes": 1,
        "save_freq": -1,
        "test_freq": -1,
        "total_epochs": 1,
        "total_training_steps": 1,
    },
}

r = requests.post(
    f"{SERVICE_URL}/api/v1/tasks",
    json={"config": config, "label": "real-grpo-1step", "priority": 10},
)
task_id = r.json()["task_id"]
print(f"Submitted: {task_id} -> {r.json()['status']}")

while True:
    r = requests.get(f"{SERVICE_URL}/api/v1/tasks/{task_id}")
    d = r.json()
    print(f"  {d['status']}", end="", flush=True)
    if d["status"] in ("completed", "failed", "cancelled"):
        print(f"\nReturn code: {d['return_code']}")
        if d["error_message"]:
            print(f"Error:\n{d['error_message']}")
        break
    time.sleep(3)
    print(".", end="", flush=True)
