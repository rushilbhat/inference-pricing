import os
import json
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from datetime import datetime, timezone

CACHE_FILE_IN = "isl_osl_timestamp_~first_1000.jsonl"
WORKLOAD_STATS_OUT = "wildchat_workload_~first_1000.json"

ISL_BINS = [0, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
OSL_BINS = ISL_BINS

# ---- Load cached 1000 rows or build them if missing ----
if os.path.exists(CACHE_FILE_IN):
    print(f"Loading cached data from {CACHE_FILE_IN}...")
    with open(CACHE_FILE_IN, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
else:
    print("Cache not found. Extracting from dataset...")

    ds = load_dataset("allenai/WildChat-1M", split="train")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    rows = []
    for entry in ds:
        if len(rows) > 1000:
            break
        conversation = entry["conversation"]
        for i in range(0, len(conversation) - 1, 2):
            if conversation[i]["role"] == "user" and conversation[i+1]["role"] == "assistant":
                ts = conversation[i+1]["timestamp"]
                isl = conversation[i]["content"]
                osl = conversation[i+1]["content"]
                isl_tokens = len(tokenizer.encode(isl))
                osl_tokens = len(tokenizer.encode(osl))
                rows.append({
                    "timestamp": ts.isoformat(),
                    "isl_tokens": isl_tokens,
                    "osl_tokens": osl_tokens,
                })

    with open(CACHE_FILE_IN, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows)} entries to {CACHE_FILE_IN}")

# ---- Build 2D histogram (counts) ----
isl_array = np.array([r["isl_tokens"] for r in rows], dtype=np.int32)
osl_array = np.array([r["osl_tokens"] for r in rows], dtype=np.int32)

H, xedges, yedges = np.histogram2d(
    isl_array,
    osl_array,
    bins=[ISL_BINS, OSL_BINS]
)
H = H.astype(int)

# ---- Joint probability matrix ----
total = H.sum()
P = (H / total).astype(float)

# ---- Sanity check: top 10 bins by probability ----
nonzero = np.argwhere(H > 0)
pairs = []
for i, j in nonzero:
    prob = float(P[i, j])
    isl_lo, isl_hi = ISL_BINS[i], ISL_BINS[i + 1]
    osl_lo, osl_hi = OSL_BINS[j], OSL_BINS[j + 1]
    pairs.append(((isl_lo, isl_hi, osl_lo, osl_hi), prob))

pairs.sort(key=lambda x: x[1], reverse=True)
print("\nTop (ISL_bin, OSL_bin) probabilities:")
for (isl_lo, isl_hi, osl_lo, osl_hi), p in pairs[:10]:
    print(f"  ISL[{isl_lo},{isl_hi}) × OSL[{osl_lo},{osl_hi}) -> {p:.4f}")

# ---- Average ISL/OSL tokens ----
isl_mids = 0.5 * (np.array(ISL_BINS[1:]) + np.array(ISL_BINS[:-1]))
osl_mids = 0.5 * (np.array(OSL_BINS[1:]) + np.array(OSL_BINS[:-1]))

P_isl = P.sum(axis=1)
P_osl = P.sum(axis=0)

avg_isl = round(float((isl_mids * P_isl).sum()))
avg_osl = round(float((osl_mids * P_osl).sum()))

print(f"\nAverage ISL (tokens) = {avg_isl}")
print(f"Average OSL (tokens) = {avg_osl}")

# ---- Arrival rate ----
ts_secs = np.array([datetime.fromisoformat(r["timestamp"]).timestamp() for r in rows], dtype=np.float64)
ts_secs.sort()
gaps = np.diff(ts_secs)

mean_gap = float(np.mean(gaps))
arrival_rate = float(1.0 / mean_gap) if mean_gap > 0 else float("inf")

print(f"\nMean inter-arrival gap: {mean_gap:.6f}s")
print(f"Arrival rate: {arrival_rate:.6f} per second")

# ---- Save workload stats ----
workload_stats = {
    "avg_isl": avg_isl,
    "avg_osl": avg_osl,

    "mean_gap_seconds": mean_gap,
    "arrival_rate_per_second": arrival_rate,

    "n_entries": len(rows),
    "window_start": datetime.fromtimestamp(ts_secs[0], timezone.utc).isoformat(),
    "window_end": datetime.fromtimestamp(ts_secs[-1], timezone.utc).isoformat()
}

with open(WORKLOAD_STATS_OUT, "w", encoding="utf-8") as f:
    json.dump(workload_stats, f, indent=2)

print(f"\nSaved workload JSON → {WORKLOAD_STATS_OUT}")
