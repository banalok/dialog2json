from datasets import load_dataset
import pandas as pd
import yaml

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
cfg = load_cfg("configs/configs.yaml")   

dataset = load_dataset(cfg["datasets"]["meddialog"]["hf_repo"], cfg["datasets"]["meddialog"]["config"], trust_remote_code=True)

# convert to pandas for easy view
df_train = dataset['train'].to_pandas()
df_val = dataset['validation'].to_pandas() if 'validation' in dataset else None
df_test = dataset['test'].to_pandas() if 'test' in dataset else None

print("Dataset shape:", df_train.shape)
print("\nColumn names:", df_train.columns.tolist())
print("\nFirst few rows:")
print(df_train.head())

# view some columns
if 'utterances' in df_train.columns:
    print("\nSample dialogue:")
    print(df_train['utterances'].iloc[0])
    
if 'dialogue_turns' in df_train.columns:
    print("\nSample dialogue turns:")
    print(df_train['dialogue_turns'].iloc[0])