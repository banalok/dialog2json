import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import os
import json
from src.chains.extractor import build_extractor, run_extractor_on_text
from src.structure.schema import validate_structured

RAW_FILE = "data/raw/meddialog_train.jsonl"
OUT_FILE = "data/structured/structured.jsonl"
LIMIT = None # change for testing lines in the jsonl file

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    chain = build_extractor()

    wrote = 0
    with open(OUT_FILE, "w", encoding="utf-8") as out_f:
        for rec in read_jsonl(RAW_FILE):
            dialogue = rec.get("dialogue_text", "")

            try:
                structured = run_extractor_on_text(chain, dialogue)
            except Exception as e:
                structured = {"conditions": [], "medications": [], "_error": str(e)}

            ok, msg = validate_structured(structured)
            if not ok:
                structured = {"conditions": [], "medications": [], "_validation": msg}

            out_line = {
                "id": rec.get("id"),                
                "description": rec.get("description", ""),
                "structured": structured
            }
            out_f.write(json.dumps(out_line, ensure_ascii=False) + "\n")
            wrote += 1
            if LIMIT and wrote >= LIMIT:
                break

    print("Wrote {} lines -> {}".format(wrote, OUT_FILE))

if __name__ == "__main__":
    main()