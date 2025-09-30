
###### manually edit conditions after generating file from this script to make it a ground truth for evaluation #########

import os, json, random
from pathlib import Path

IN_FILE  = "data/structured/structured.jsonl"
OUT_FILE = "data/structured/data_eval.jsonl"
N = 20   # data points num
SEED = 42

def read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    rows = list(read_jsonl(IN_FILE))
    random.Random(SEED).shuffle(rows)
    rows = rows[:N]

    with open(OUT_FILE, "w", encoding="utf-8") as out:
        for r in rows:            
            pred_names = [ (c.get("name") or "").strip() for c in r.get("structured",{}).get("conditions",[]) ]
            true_line = {
                "id": r.get("id"),
                "conditions_eval": pred_names,                  
            }
            out.write(json.dumps(true_line, ensure_ascii=False) + "\n")

    print("Wrote template:", Path(OUT_FILE).resolve())

