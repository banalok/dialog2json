import json

PRED_FILE = "data/structured/structured.jsonl"   # model outputs
GOLD_FILE = "data/structured/data_eval.jsonl"         #  corrected truth

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def norm(s):
    return (s or "").lower().strip()

# compare based on id
pred = {}
for rec in read_jsonl(PRED_FILE):
    names = []
    st = rec.get("structured", {}) or {}
    for c in st.get("conditions", []):
        n = norm(c.get("name"))
        if n:
            names.append(n)
    pred[rec.get("id")] = set(names)


true = {}
for rec in read_jsonl(GOLD_FILE):
    names = [norm(x) for x in rec.get("conditions_eval", []) if norm(x)]
    true[rec.get("id")] = set(names)

missing = [i for i in true if i not in pred]
if missing:
    print("WARNING: true IDs missing in predictions (showing up to 5):", missing[:5])

# metrics
tp = fp = fn = 0
for i, true_set in true.items():
    pred_set = pred.get(i, set())
    tp += len(pred_set & true_set)
    fp += len(pred_set - true_set)
    fn += len(true_set - pred_set)

prec = tp / (tp + fp) if (tp + fp) else 0.0
rec  = tp / (tp + fn) if (tp + fn) else 0.0
f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

print("TP:", tp, " FP:", fp, " FN:", fn)
print("Precision:", round(prec, 3))
print("Recall:",    round(rec, 3))
print("F1:",        round(f1, 3))
