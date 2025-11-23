from datetime import date

SERVER_TYPE = "NVIDIA_DGX_Station_A100_80GB"
SERVER_COST_PER_HR = 5.00 

PORT = 8000

TARGET_UTILISATION = 1.00

TRAFFIC_START_DATE = date(2024, 3, 17)
TRAFFIC_END_DATE = date(2024, 3, 23)

MODEL_FAMILY = "Meta-Llama-3.1-8B-Instruct"
QUANTIZATION_MAP = {
    "base":  f"unsloth/{MODEL_FAMILY}",
    "w8a16": f"RedHatAI/{MODEL_FAMILY}-quantized.w8a16",
    "w8a8":  f"RedHatAI/{MODEL_FAMILY}-quantized.w8a8",
    "w4a16": f"RedHatAI/{MODEL_FAMILY}-quantized.w4a16",
}

SLO_LEVELS = [
    ("Baseline", 1.0,  0.05),   # 20 tok/s
    ("Fast",     0.5,  0.0167), # 60 tok/s
    ("Instant",  0.25, 0.01),   # 100 tok/s
]

RESULTS_FOLDER = "benchmark_results"
RESULTS_FILENAME = f"{RESULTS_FOLDER}/{MODEL_FAMILY}_{SERVER_TYPE}.jsonl"