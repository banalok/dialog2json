
import os
import json
from pathlib import Path

CATEGORIES = {
    "respiratory": [
        "cough", "dry cough", "shortness of breath", "sob",
        "chest tightness", "influenza", "cold", "upper respiratory infection"
    ],
    "ent": [
        "sore throat", "pharyngitis", "strep throat", "throat pain"
    ],
    "neuro": [
        "dizziness", "headache", "migraine", "vertigo"
    ],
    "general": [
        "fever", "fatigue"
    ],
}

def categorize(name):
    n = (name or "").lower().strip() 
    for cat, terms in CATEGORIES.items():
        if n in terms:
            return cat
    return "other"


IN_FILE  = "data/structured/structured.jsonl"
OUT_FILE = "data/structured/structured_tagged.jsonl"

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    n = 0
    with open(OUT_FILE, "w", encoding="utf-8") as out_f:
        for rec in read_jsonl(IN_FILE):
            st = rec.get("structured", {}) or {}
            tagged_conditions = []
            for c in st.get("conditions", []):
                name = (c.get("name") or "").strip()
                tagged_conditions.append({
                    "name": name,
                    "category": categorize(name)
                })
            out_line = {
                "id": rec.get("id"),
                "structured": st,
                "tagged": {
                    "conditions": tagged_conditions,
                    "medications": st.get("medications", [])
                }
            }
            out_f.write(json.dumps(out_line, ensure_ascii=False) + "\n")
            n += 1
    print("Wrote {} lines -> {}".format(n, Path(OUT_FILE).resolve()))

if __name__ == "__main__":
    main()
