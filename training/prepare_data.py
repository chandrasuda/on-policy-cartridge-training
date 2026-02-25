#!/usr/bin/env python3
"""
Prepare LongHealth patient questions as veRL training data.

Downloads the LongHealth benchmark, formats the questions as prompts,
and saves them as parquet files that veRL's data pipeline can read.

Usage:
    python examples/cartridge_distill/prepare_data.py

Output:
    ~/data/cartridge_distill/train.parquet
    ~/data/cartridge_distill/val.parquet
"""

import json
import os
from pathlib import Path

import pandas as pd
import requests


def load_longhealth():
    """Download and parse the LongHealth benchmark.

    Each row includes:
      - prompt: the question in chat format (for the student rollout)
      - document_text: the full patient documents (for the teacher context)
      - patient_id, correct_answer: metadata
    """
    url = "https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json"
    print(f"Downloading LongHealth from {url}...")
    data = requests.get(url).json()

    prompts = []
    for patient_id, patient in data.items():
        name = patient["name"]
        diagnosis = patient["diagnosis"]
        birthday = patient["birthday"]

        # Concatenate ALL patient documents into one string for teacher context
        document_text = "\n\n".join(
            f"--- {doc_id} ---\n{doc_text}"
            for doc_id, doc_text in patient["texts"].items()
        )

        for q in patient["questions"]:
            options = (
                f"A) {q['answer_a']}\n"
                f"B) {q['answer_b']}\n"
                f"C) {q['answer_c']}\n"
                f"D) {q['answer_d']}\n"
                f"E) {q['answer_e']}"
            )
            prompt = (
                f"You are answering questions about a patient. "
                f"Patient: {name}, DOB: {birthday}, Diagnosis: {diagnosis}.\n\n"
                f"Question: {q['question']}\n\n"
                f"Options:\n{options}\n\n"
                f"Think step by step, then give your final answer."
            )
            prompts.append({
                "prompt": [{"role": "user", "content": prompt}],
                "document_text": document_text,
                "data_source": "longhealth",
                "patient_id": patient_id,
                "correct_answer": q["correct"],
                # veRL's reward loop expects this field
                "reward_model": {"ground_truth": q["correct"], "style": "rule"},
            })

    return prompts


def main():
    prompts = load_longhealth()
    print(f"Total prompts: {len(prompts)}")

    # Split: first 8 patients for train, last 2 for val
    train = [p for p in prompts if int(p["patient_id"].split("_")[1]) <= 8]
    val = [p for p in prompts if int(p["patient_id"].split("_")[1]) > 8]
    print(f"Train: {len(train)}, Val: {len(val)}")

    out_dir = Path.home() / "data" / "cartridge_distill"
    out_dir.mkdir(parents=True, exist_ok=True)

    # veRL expects parquet with at minimum a "prompt" column
    # prompt should be a list of message dicts (chat format)
    train_df = pd.DataFrame(train)
    val_df = pd.DataFrame(val)

    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"

    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)

    print(f"\n✓ Saved {train_path} ({len(train_df)} rows)")
    print(f"✓ Saved {val_path} ({len(val_df)} rows)")
    print(f"\nSample prompt:\n{train_df.iloc[0]['prompt']}")


if __name__ == "__main__":
    main()
