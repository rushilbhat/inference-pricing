import os
import json
import numpy as np
from datasets import load_dataset
from datetime import datetime

class TrafficProfile:

    BINS = [0, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    def __init__(self, tokenizer, cache_file="wildchat_cache.jsonl", limit=1000):
        self.tokenizer = tokenizer
        self.cache_file = cache_file
        self.limit = limit
        self.entries = self._load_entries()

    def _load_entries(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f]

        ds = load_dataset("allenai/WildChat-1M", split="train")
        
        rows = []
        for entry in ds:
            if len(rows) >= self.limit:
                break
            
            conv = entry["conversation"]
            history_len = 0
            for i in range(0, len(conv) - 1, 2):
                user_msg = conv[i]
                asst_msg = conv[i+1]
                if user_msg["role"] == "user" and asst_msg["role"] == "assistant":
                    user_len = len(self.tokenizer.encode(user_msg["content"]))
                    asst_len = len(self.tokenizer.encode(asst_msg["content"]))

                    rows.append({
                        "timestamp": asst_msg["timestamp"].isoformat(),
                        "isl": history_len + user_len,
                        "osl": asst_len,
                    })
                    
                    history_len += (user_len + asst_len)

                    if len(rows) >= self.limit: 
                        break

        with open(self.cache_file, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        
        print(f"[TrafficProfile] Cached {len(rows)} entries.")
        return rows

    def compute_metrics(self):
        isl_array = np.array([r["isl"] for r in self.entries], dtype=np.int32)
        osl_array = np.array([r["osl"] for r in self.entries], dtype=np.int32)

        H, _, _ = np.histogram2d(
            isl_array, 
            osl_array, 
            bins=[self.BINS, self.BINS]
        )
        total_count = H.sum()
        P = H / total_count

        P_isl = P.sum(axis=1)
        P_osl = P.sum(axis=0)
        midpoints = 0.5 * (np.array(self.BINS[1:]) + np.array(self.BINS[:-1]))
        w_avg_isl = np.sum(midpoints * P_isl)
        w_avg_osl = np.sum(midpoints * P_osl)

        timestamps = sorted([
            datetime.fromisoformat(r["timestamp"]).timestamp() 
            for r in self.entries
        ])
        gaps = np.diff(timestamps)
        mean_gap = np.mean(gaps)
        arrival_rate = 1.0 / mean_gap

        return {
            "avg_isl": int(round(w_avg_isl)),
            "avg_osl": int(round(w_avg_osl)),
            "arrival_rate": arrival_rate
        }
