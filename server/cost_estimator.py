import os
import random
import logging
import pandas as pd
from random import sample
from dotenv import load_dotenv

from RAGHelper_local import RAGHelperLocal
from RAGHelper import RAGHelper




LLM_GPU_REQUIREMENTS = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {"gpu": "A6000", "gpu_fraction": 1.0},
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {"gpu": "A6000", "gpu_fraction": 1},
    "Qwen/Qwen2.5-7B-Instruct": {"gpu": "A6000", "gpu_fraction": 1},
    "Qwen/Qwen3-8B": {"gpu": "A6000", "gpu_fraction": 1},
    "tiiuae/Falcon3-7B-Instruct": {"gpu": "A6000", "gpu_fraction": 1},
    "meta-llama/Llama-3.2-3B-Instruct": {"gpu": "A5000", "gpu_fraction": 0.5},
    "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf": {"gpu": "A5000", "gpu_fraction": 0.5},
}

GPU_MONTHLY_COST = {
    "A100": 1500,
    "A6000": 700,
    "A5000": 375,
    "L4": 300,
}

def estimate_cost_per_model(model_name):
    gpu_info = LLM_GPU_REQUIREMENTS.get(model_name)
    if not gpu_info:
        return None
    gpu_type = gpu_info["gpu"]
    gpu_fraction = gpu_info["gpu_fraction"]
    total_month_hours = 720
    active_hours = 5 * 4 * 8
    gpu_cost_scaled = (active_hours / total_month_hours) * GPU_MONTHLY_COST[gpu_type] * gpu_fraction
    cpu_cost = 8 * 12.50
    ram_cost = 64 * 6.00
    disk_cost = 500 * 0.16
    infra_cost = cpu_cost + ram_cost + disk_cost
    return round(infra_cost + gpu_cost_scaled, 2)