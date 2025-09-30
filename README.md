This repo aims to turn raw patient–doctor chats into clean training data, step by step. 
- First, it loads the MedDialog-EN raw text from HuggingFace (https://huggingface.co/datasets/bigbio/meddialog) and saves to JSONL.
- Second, a LangChain extractor converts each dialogue into structured JSON with only two keys: conditions and medications.
- Third, a tiny cleaner normalizes names and removes labs/normal findings; a tagger adds simple categories.
- Fourth, a script builds chat-format fine-tuning pairs (system + user dialogue → assistant JSON) ready for OpenAI Supervised Fine-tuning (SFT).

Check data/raw/meddialog_train.jsonl (raw) against data/finetune/train.jsonl to compare raw text with model-ready data.
