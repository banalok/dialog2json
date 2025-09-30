import os
import json
from datasets import load_dataset
import yaml
from pathlib import Path

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_meddialog(repo, config, split_train, split_val, split_test):
    ds = load_dataset(repo, config, trust_remote_code=True)
    return ds[split_train], ds[split_val], ds[split_test]

def format_dialog(utterances):
    return "\n".join(utterances)

def to_records(split, n=None):
    if n is None:
        n = len(split)
    recs = []
    for i in range(min(n, len(split))):
        item = split[i]
        utter = item.get("utterances") or item.get("dialogue_turns")
        dialogue_text = format_dialog(utter)
        recs.append({
            "id": "meddialog_{}".format(i),
            "description": item.get("description", ""),
            "utterances": utter,
            "dialogue_text": dialogue_text
        })
    return recs

def save_jsonl(records, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def sample_and_cache(cfg):
    raw_dir = cfg["raw_dir"]
    os.makedirs(raw_dir, exist_ok=True)

    train, val, test = load_meddialog(
        cfg["datasets"]["meddialog"]["hf_repo"],
        cfg["datasets"]["meddialog"]["config"],
        cfg["datasets"]["meddialog"]["split_train"],
        cfg["datasets"]["meddialog"]["split_val"],
        cfg["datasets"]["meddialog"]["split_test"],
    )

    train_recs = to_records(train, cfg["sampling"]["train_n"])
    val_recs   = to_records(val,   cfg["sampling"]["val_n"])
    test_recs  = to_records(test,  cfg["sampling"]["test_n"])

    paths = {
        "train": os.path.join(raw_dir, "meddialog_train.jsonl"),
        "val":   os.path.join(raw_dir, "meddialog_val.jsonl"),
        "test":  os.path.join(raw_dir, "meddialog_test.jsonl"),
    }
    save_jsonl(train_recs, paths["train"])
    save_jsonl(val_recs,   paths["val"])
    save_jsonl(test_recs,  paths["test"])
    return paths

##### run it once to cache data to local wihtout needing to call HF repeatedly ###############################

if __name__ == "__main__":
    cfg = load_cfg("configs/configs.yaml")
    paths = sample_and_cache(cfg)
    print("Saved subsets to:")
    for k, v in paths.items():
        print("  {}: {}".format(k, Path(v).resolve()))
