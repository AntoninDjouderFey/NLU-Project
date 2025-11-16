#!/usr/bin/env python3
# coding: utf-8

import os
import json
import csv
import re
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import argparse
#auto tokeniseur allow to use multiple models while keeping the same code 
from transformers.models.auto.tokenization_auto import AutoTokenizer
import math

# ---------- Arguments ----------
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default="../massive_extract", help="Directory with .jsonl files")
parser.add_argument("--out_dir", default="csv", help="Directory to save CSV outputs")
parser.add_argument("--tokenizer", default="bert-base-multilingual-cased",
                    help="HF tokenizer name (use a multilingual tokenizer for zero-shot)")
parser.add_argument("--languages_exlude", nargs="*", default=[], help="Locales to exclude from training (e.g. fr-FR fr-CA)")
parser.add_argument("--train_val_split", type=float, default=0.1, help="If 'valid' partition not present, split train -> valid fraction")
parser.add_argument("--min_slot_coverage", type=float, default=0.0, help="If >0: filter utterances that have no slot labels")
args = parser.parse_args()

input_dir = args.input_dir
out_dir = args.out_dir
tokenizer_name = args.tokenizer
languages_exlude = set(args.languages_exlude)
VALID_SPLIT = args.train_val_split
MIN_SLOT_COVERAGE = args.min_slot_coverage

os.makedirs(out_dir, exist_ok=True)

# ---------- Tokenizer ----------
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
if not tokenizer.is_fast:
    raise ValueError("I need a tokeniser with offset mapping.")

SLOT_PATTERN = re.compile(r"\[([^\s\[\]:]+)\s*:\s*([^\]]+)\]")  # matches [slot : value] 

def parse_annot_utt(annot_utt: str) -> List[Tuple[str, str]]:
    """
    Returns list of (slot_type, slot_text) extracted from str.
    """
    slots = []
    for m in SLOT_PATTERN.finditer(annot_utt):
        slot_type = m.group(1).strip()
        slot_text = m.group(2).strip()
        slots.append((slot_type, slot_text))
    return slots

def slots_to_bio_list(annot_utt: str, text: str) -> List[Tuple[str,str]]:
    """
    Parse annot_utt and create ordered list of (slot_type, slot_text).
    """
    parsed = parse_annot_utt(annot_utt)
    return parsed

def find_slot_spans_in_text(text: str, slot_texts: List[str]) -> List[Tuple[int,int]]:
    """
    Given `text` and a list of slot_text occurrences in textual order,
    return a list of (start_char, end_char) for each slot_text.
    This function searches progressively to handle repeated values.
    """
    spans = []
    search_start = 0
    for s in slot_texts:
        if not s:
            spans.append((-1,-1))
            continue
        idx = text.find(s, search_start)
        if idx == -1:
            # fallback: try full-text search from beginning (may happen due to punctuation or tokenization differences)
            idx = text.find(s)
        if idx == -1:
            spans.append((-1,-1))
        else:
            spans.append((idx, idx + len(s)))
            search_start = idx + len(s)
    return spans

def align_tokens_and_labels(text: str, slots: List[Tuple[str,str]]) -> Tuple[List[str], List[str]]:
    """
    Tokenize `text` with tokenizer and produce BIO labels aligned to tokens.
    Returns (tokens, bio_labels) where tokens are tokenizer tokens (strings).
    Ps: This part is inpired by differents github projects 
    """
    if len(slots) == 0:
        #tokenize and return all 'O'
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
        return tokens, ["O"] * len(tokens)

    # list of slot texts (in same order) and slot types
    slot_types = [t for t, _ in slots]
    slot_texts = [v for _, v in slots]

    # find spans (char-level) in the original text for each slot_text (progressive search)
    spans = find_slot_spans_in_text(text, slot_texts)

    # build list of slot spans with types
    slot_spans = []
    for (stype, stext), (sstart, send) in zip(slots, spans):
        if sstart == -1:
            print("How did we end up here?")
            continue
        slot_spans.append({"type": stype, "start": sstart, "end": send, "text": stext})

    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]   
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])

    bio = ["O"] * len(tokens)

    # assign BIO labels by checking token offset overlap with slot spans
    for s in slot_spans:
        sstart, send, stype = s["start"], s["end"], s["type"]
        first = True
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_end <= tok_start:
                continue
            if tok_end <= sstart or tok_start >= send:
                continue
            if first:
                bio[i] = f"B-{stype}"
                first = False
            else:
                bio[i] = f"I-{stype}"

    return tokens, bio


intent_rows = []
slot_rows = []
slot_label_set = set()
missing_slot_loc_count = 0
total_rows = 0

for fname in sorted(os.listdir(input_dir)):
    if not fname.endswith(".jsonl"):
        continue
    path = os.path.join(input_dir, fname)
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            total_rows += 1
            data = json.loads(line)
            text = data.get("utt", "")
            intent = data.get("intent", "")
            partition = data.get("partition", "train")
            locale = data.get("locale", "unknown")
            annot_utt = data.get("annot_utt", "")

            if not text or not intent:
                continue
            if locale in languages_exlude:
                continue

            # intent row
            intent_rows.append({
                "text": text,
                "label": intent,
                "partition": partition,
                "locale": locale
            })

            # slots: only if annot_utt present
            if annot_utt and annot_utt.strip():
                slots = slots_to_bio_list(annot_utt, text)
                tokens, bio = align_tokens_and_labels(text, slots)

                # collect slot label types
                for b in bio:
                    slot_label_set.add(b)

                # optionally filter utterances with no slot coverage
                n_non_o = sum(1 for b in bio if b != "O")
                if MIN_SLOT_COVERAGE > 0.0:
                    if (n_non_o / max(1, len(bio))) < MIN_SLOT_COVERAGE:
                        continue

                slot_rows.append({
                    "text": text,
                    "tokens": " ".join(tokens),
                    "labels": " ".join(bio),
                    "partition": partition,
                    "locale": locale,
                    "intent": intent
                })


print(f"[INFO] processed {total_rows} jsonl lines -> {len(intent_rows)} intent rows, {len(slot_rows)} slot rows")
# ensure consistent ordering for slot labels (O,B,I)
slot_labels_sorted = sorted(list(slot_label_set), key=lambda x: (0 if x=="O" else 1, x))

# just in case 
#if "O" in slot_labels_sorted:
#    slot_labels_sorted.remove("O")
#    slot_labels_sorted = ["O"] + slot_labels_sorted


intent_csv = os.path.join(out_dir, "nlu_intent_all.csv")
slots_csv = os.path.join(out_dir, "nlu_slots_all.csv")
slot_labels_json = os.path.join(out_dir, "slot_labels.json")

# intents
with open(intent_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["text", "label", "partition", "locale"])
    writer.writeheader()
    for r in intent_rows:
        writer.writerow(r)

# slots
with open(slots_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["text", "tokens", "labels", "partition", "locale", "intent"])
    writer.writeheader()
    for r in slot_rows:
        writer.writerow(r)

# slot label 
with open(slot_labels_json, "w", encoding="utf-8") as f:
    json.dump({"labels": slot_labels_sorted}, f, ensure_ascii=False, indent=2)

print(f"[INFO] saved intent CSV: {intent_csv}")
print(f"[INFO] saved slots CSV:  {slots_csv}")
print(f"[INFO] saved slot label list ({len(slot_labels_sorted)}): {slot_labels_json}")
print(f"[INFO] slot labels: {slot_labels_sorted}")



def split_and_write(csv_in_path, out_prefix):
    """
    Read CSV with 'partition' column and write three files:
    out_prefix_train.csv, out_prefix_valid.csv, out_prefix_test.csv
    If no 'valid' rows present, will split 'train' into train+valid using VALID_SPLIT.
    """
    import pandas as pd
    df = pd.read_csv(csv_in_path)
    parts = df['partition'].unique().tolist()
    if 'valid' in parts:
        df[df['partition']=='train'].to_csv(f"{out_prefix}_train.csv", index=False)
        df[df['partition']=='valid'].to_csv(f"{out_prefix}_valid.csv", index=False)
        df[df['partition']=='test'].to_csv(f"{out_prefix}_test.csv", index=False)
    else:
        raise ZeroDivisionError("Not a zero division in reality just a bad set of data")
        train_df = df[df['partition']=='train'].sample(frac=1.0, random_state=42).reset_index(drop=True)
        n_val = int(math.ceil(len(train_df) * VALID_SPLIT))
        val_df = train_df.iloc[:n_val]
        tr_df = train_df.iloc[n_val:]
        tr_df.to_csv(f"{out_prefix}_train.csv", index=False)
        val_df.to_csv(f"{out_prefix}_valid.csv", index=False)
        if 'test' in parts:
            df[df['partition']=='test'].to_csv(f"{out_prefix}_test.csv", index=False)
        else:
            print(f"[WARN] no 'test' partition found in {csv_in_path}; not writing test split.")

# create splits for both
split_and_write(intent_csv, os.path.join(out_dir, "intent"))
split_and_write(slots_csv, os.path.join(out_dir, "slots"))

print("Preprocessing finished.")
