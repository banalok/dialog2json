import os, json, random
from pathlib import Path

RAW_FILE  = "data/raw/meddialog_train.jsonl"        # has id and dialogue_text
PRED_FILE = "data/structured/structured.jsonl"      # current extractor outputs
OUT_DIR   = "data/finetune"
TRAIN_OUT = f"{OUT_DIR}/train.jsonl"
VALID_OUT = f"{OUT_DIR}/valid.jsonl"
VALID_FRAC = 0.1
SEED = 7

SYSTEM_PROMPT = (
    "You are a clinical information extractor. Produce strict JSON with exactly two keys: "
    "\"conditions\" and \"medications\". Follow this schema:\n"
    "- conditions: [{\"name\" (required), \"status\" (optional: active/suspected/possible/resolved), "
    "\"onset\" (optional), \"negated\" (optional boolean)}]\n"
    "- medications: [{\"name\" (required), \"dose\"?, \"route\"?, \"freq\"?, \"negated\"?}]\n"
    "Rules: Return ONLY JSON. No extra text."
)

def read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def map_by_id(path):
    m = {}
    for r in read_jsonl(path):
        m[r["id"]] = r
    return m

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raw = map_by_id(RAW_FILE)
    pred = map_by_id(PRED_FILE)

    pairs = []
    for _id, raw_row in raw.items():
        if _id not in pred:
            continue
        dialogue = raw_row.get("dialogue_text") or ""
        structured = pred[_id].get("structured") or {"conditions": [], "medications": []}        
        ex = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Dialogue:\n{dialogue}\n\nReturn only the JSON object."},
                {"role": "assistant", "content": json.dumps(structured, ensure_ascii=False)}
            ]
        }
        pairs.append(ex)

    random.Random(SEED).shuffle(pairs)
    n_valid = max(1, int(len(pairs) * VALID_FRAC))
    valid = pairs[:n_valid]
    train = pairs[n_valid:]

    with open(TRAIN_OUT, "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(VALID_OUT, "w", encoding="utf-8") as f:
        for r in valid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train)} train and {len(valid)} valid → {Path(OUT_DIR).resolve()}")

if __name__ == "__main__":
    main()
