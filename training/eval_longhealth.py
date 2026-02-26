#!/usr/bin/env python3
"""
Evaluate LongHealth multiple-choice accuracy via Tokasaurus.

Compares:
  1. No cartridge (baseline)
  2. Pre-trained cartridge (hazyresearch/cartridge-wauoq23f)

Usage:
    python eval_longhealth.py
"""

import json
import requests
import re
import time

TOKASAURUS_URL = "https://kiran1234c--tokasaurus-cartridge-server-serve.modal.run"

# Load LongHealth questions
print("Loading LongHealth benchmark...")
data = requests.get(
    "https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json"
).json()

questions = []
for pid, patient in data.items():
    for q in patient["questions"]:
        options = (
            f"A) {q['answer_a']}\n"
            f"B) {q['answer_b']}\n"
            f"C) {q['answer_c']}\n"
            f"D) {q['answer_d']}\n"
            f"E) {q['answer_e']}"
        )
        prompt = (
            f"You are answering a multiple choice question about patient {patient['name']}.\n\n"
            f"Question: {q['question']}\n\n"
            f"Options:\n{options}\n\n"
            f"Answer with ONLY the letter (A, B, C, D, or E):"
        )
        # Map correct answer text to letter
        answer_map = {
            q["answer_a"]: "A", q["answer_b"]: "B", q["answer_c"]: "C",
            q["answer_d"]: "D", q["answer_e"]: "E",
        }
        correct_letter = answer_map.get(q["correct"], "?")
        
        questions.append({
            "prompt": prompt,
            "correct": correct_letter,
            "correct_text": q["correct"],
            "patient_id": pid,
            "patient_name": patient["name"],
        })

# Only eval first 8 patients (training set) — 160 questions
train_questions = [q for q in questions if int(q["patient_id"].split("_")[1]) <= 8]
print(f"Total questions: {len(questions)}, Training patients (1-8): {len(train_questions)}")

# Use a subset for speed
eval_qs = train_questions[:40]  # 40 questions


def extract_answer(text):
    """Extract A/B/C/D/E from model output."""
    text = text.strip()
    # Try to find a letter at the start
    match = re.search(r'\b([A-E])\b', text[:20])
    if match:
        return match.group(1)
    # Try to match the full answer text
    for letter in "ABCDE":
        if text.startswith(f"{letter})") or text.startswith(letter):
            return letter
    return text[:1].upper() if text else "?"


def evaluate(cartridges=None, label=""):
    """Run evaluation with optional cartridge."""
    correct = 0
    total = 0
    errors = 0

    for i, q in enumerate(eval_qs):
        payload = {
            "model": "default",
            "prompt": q["prompt"],
            "max_tokens": 10,
            "temperature": 0.0,
        }
        if cartridges:
            payload["cartridges"] = cartridges

        try:
            resp = requests.post(
                f"{TOKASAURUS_URL}/custom/cartridge/completions",
                json=payload,
                timeout=120,
            )
            if resp.status_code == 200:
                result = resp.json()
                answer_text = result["choices"][0]["text"]
                predicted = extract_answer(answer_text)
                expected = extract_answer(q["correct"])

                is_correct = predicted == expected
                if is_correct:
                    correct += 1
                total += 1

                if i < 5 or (not is_correct and i < 20):
                    mark = "✓" if is_correct else "✗"
                    print(f"  {mark} Q{i}: predicted={predicted}, expected={expected}, text='{answer_text.strip()[:50]}'")
            else:
                errors += 1
                if errors <= 3:
                    print(f"  ⚠ HTTP {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  ⚠ Error: {e}")

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n{label}: {correct}/{total} correct ({accuracy:.1f}%) [{errors} errors]")
    return accuracy


# Warm up Tokasaurus
print("\nWarming up Tokasaurus...")
try:
    requests.get(f"{TOKASAURUS_URL}/ping", timeout=300)
    print("✓ Tokasaurus ready\n")
except:
    print("⚠ Tokasaurus may be cold-starting...\n")

# Eval 1: No cartridge (baseline)
print("=" * 50)
print("EVAL 1: No cartridge (baseline)")
print("=" * 50)
baseline_acc = evaluate(cartridges=None, label="Baseline")

# Eval 2: Pre-trained cartridge
print("\n" + "=" * 50)
print("EVAL 2: Pre-trained cartridge (hazyresearch/cartridge-wauoq23f)")
print("=" * 50)
cartridge_acc = evaluate(
    cartridges=[{"id": "hazyresearch/cartridge-wauoq23f", "source": "huggingface"}],
    label="Cartridge",
)

# Summary
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"Baseline (no cartridge):    {baseline_acc:.1f}%")
print(f"Pre-trained cartridge:      {cartridge_acc:.1f}%")
print(f"Improvement:                {cartridge_acc - baseline_acc:+.1f}%")
