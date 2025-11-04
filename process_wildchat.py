import os
import json
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from datetime import datetime, timezone

CACHE_FILE_IN = "isl_osl_timestamp_~first_1000.jsonl"
CACHE_FILE_PROBS = "isl_osl_jointprobs_~first_1000.json"
ARRIVAL_OUT = "wildchat_arrival_~first_1000.json"

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

probs_payload = {
    "isl_bins": ISL_BINS,
    "osl_bins": OSL_BINS,
    "probabilities": P.tolist()
}

with open(CACHE_FILE_PROBS, "w", encoding="utf-8") as f:
    json.dump(probs_payload, f, indent=2)

print(f"Joint probability matrix saved to {CACHE_FILE_PROBS}")

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

# ---- Arrival rate and burstiness ----
ts_secs = np.array([datetime.fromisoformat(r["timestamp"]).timestamp() for r in rows], dtype=np.float64)
ts_secs.sort()
gaps = np.diff(ts_secs)

mean_gap = float(np.mean(gaps))
std_gap = float(np.std(gaps))
cv = std_gap / mean_gap
k = 1.0 / (cv ** 2)
theta = mean_gap / k

arrival_stats = {
    "k": k,
    "theta": theta,
    "n_entries": int(len(ts_secs)),
    "window": {
        "start": datetime.fromtimestamp(ts_secs[0], timezone.utc).isoformat(),
        "end": datetime.fromtimestamp(ts_secs[-1], timezone.utc).isoformat()
    }
}

with open(ARRIVAL_OUT, "w", encoding="utf-8") as f:
    json.dump(arrival_stats, f, indent=2)

print(f"\nArrival params saved to {ARRIVAL_OUT}")
print(f"  k     : {k:.3f}")
print(f"  theta : {theta:.6f} s")
