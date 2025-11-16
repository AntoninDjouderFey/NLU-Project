#!/usr/bin/env python3
# coding: utf-8
"""
data_loader.py

Fournit :
 - NLUdataset : lit un CSV de slots (qui contient texte, tokens, labels, partition, locale, intent)
 - DataCollatorForNLU : regroupe des batches, remplit les entrées, convertit les labels de slots en ids et remplit avec -100

Hypothèses :
 - Les CSV ont été produits par le script de prétraitement et incluent les colonnes :
    text, tokens, labels, partition, locale, intent
 - Le tokenizer utilisé pour le prétraitement doit être le même que celui passé ici (sino ca ne marche pas).
"""

import json
import os
import csv
from typing import List, Dict, Tuple, Optional
import pandas as pd
from transformers.tokenization_utils import PreTrainedTokenizer
import torch
from torch.utils.data import Dataset
import numpy as np

class NLUdataset(Dataset):
    """
    Lit un CSV de slots et prépare des exemples pour l'entraînement conjoint intent+slot.
    Chaque exemple contient :
        - input_ids, attention_mask (non remplis)
        - intent_label
        - slot_labels (liste de chaînes BIO alignées sur les tokens)
    """
    def __init__(self,
                 csv_path: str,
                 tokenizer: PreTrainedTokenizer,
                 slot_label_list: Optional[List[str]] = None,
                 intent_label_list: Optional[List[str]] = None,
                 use_tokens_column: bool = True):
        print(f"[NLUdataset] Chargement du CSV depuis {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.use_tokens_column = use_tokens_column

        # La toute première version n'utilisait pas de pretraining à part 
        if slot_label_list is None:
            print("Déduction des labels de slots depuis le CSV")
            s = set()
            for labstr in self.df['labels'].fillna(""):
                for t in str(labstr).split():
                    s.add(t)
            slot_label_list = sorted(list(s), key=lambda x: (0 if x == "O" else 1, x))
        if intent_label_list is None:
            print("Déduction des labels d'intent depuis le CSV")
            intent_label_list = sorted(self.df['intent'].unique().tolist())

        self.slot_label_list = slot_label_list
        self.intent_label_list = intent_label_list

        self.slot_label2id = {lab: i for i, lab in enumerate(self.slot_label_list)}
        self.intent_label2id = {lab: i for i, lab in enumerate(self.intent_label_list)}

        # construire la liste des exemples à conserver
        print("Construction des exemples à partir du CSV")
        self.examples = []
        for i, row in self.df.iterrows():
            text = str(row['text'])
            intent = row['intent']
            partition = row.get('partition', None)
            tokens_col = str(row.get('tokens', "")).strip()
            labels_col = str(row.get('labels', "")).strip()

            # liste de labels
            labels_list = labels_col.split() if labels_col else []

            # essayer de tokenizer le texte avec le tokenizer
            enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            tok_offsets = enc.get('offset_mapping', [])
            tok_tokens = self.tokenizer.convert_ids_to_tokens(enc['input_ids'])

            if len(tok_tokens) == len(labels_list):
                tokens_for_storage = tok_tokens
                labels_for_storage = labels_list
                #print(f"[NLUdataset] Exemple {i}: tokenisation OK")
            else:
                if tokens_col and self.use_tokens_column:
                    tok_tokens = tokens_col.split()
                    if len(tok_tokens) == len(labels_list):
                        tokens_for_storage = tok_tokens
                        labels_for_storage = labels_list
                    else:
                        if len(labels_list) < len(tok_tokens):
                            labels_for_storage = labels_list + ["O"] * (len(tok_tokens) - len(labels_list))
                            tokens_for_storage = tok_tokens
                        else:
                            labels_for_storage = labels_list[:len(tok_tokens)]
                            tokens_for_storage = tok_tokens
                else:
                    if len(labels_list) < len(tok_tokens):
                        labels_for_storage = labels_list + ["O"] * (len(tok_tokens) - len(labels_list))
                    else:
                        labels_for_storage = labels_list[:len(tok_tokens)]
                    tokens_for_storage = tok_tokens

            self.examples.append({
                "text": text,
                "tokens": tokens_for_storage,
                "labels": labels_for_storage,
                "intent": intent,
                "partition": partition
            })

        print(f" {len(self.examples)} exemples construits")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        # convertir les tokens -> ids
        try:
            input_ids = self.tokenizer.convert_tokens_to_ids(ex['tokens'])
        except Exception:
            enc = self.tokenizer(ex['text'], add_special_tokens=False)
            input_ids = enc['input_ids']
            print(f" item {idx}: re-tokenisation utilisée")

        # mapper les labels de slots en ids
        slot_label_ids = [self.slot_label2id.get(l, self.slot_label2id.get("O", 0)) for l in ex['labels']]
        intent_id = self.intent_label2id.get(ex['intent'], 0)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "slot_labels": torch.tensor(slot_label_ids, dtype=torch.long),
            "intent_label": torch.tensor(intent_id, dtype=torch.long),
            "tokens": ex['tokens'],
            "text": ex['text']
        }

class DataCollatorForNLU:
    """
    Regroupe un batch de NLUdataset :
        - remplit input_ids via tokenizer.pad,
        - remplit slot_labels avec -100,
        - retourne des tenseurs.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, pad_to_multiple_of: Optional[int] = None, slot_pad_token_id: int = -100):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.slot_pad_token_id = slot_pad_token_id

    def __call__(self, features: List[Dict]):
        #print(f"Collating {len(features)} features")
        input_ids_list = [f['input_ids'] for f in features]
        batch_enc = {"input_ids": [[int(token) for token in x.cpu().numpy()] for x in input_ids_list]}
        padded = self.tokenizer.pad(
            batch_enc,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of
        )

        max_len = padded['input_ids'].size(1)
        slot_labels_padded = []
        for i, f in enumerate(features):
            lbls = f['slot_labels'].tolist()
            if len(lbls) < max_len:
                padded_lbls = lbls + [self.slot_pad_token_id] * (max_len - len(lbls))
            else:
                padded_lbls = lbls[:max_len]
            slot_labels_padded.append(padded_lbls)
            #print(f"Feature {i}: slot_labels padded to {len(padded_lbls)}")

        slot_labels = torch.tensor(slot_labels_padded, dtype=torch.long)
        intent_labels = torch.tensor([f['intent_label'].item() for f in features], dtype=torch.long)

        batch = {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "slot_labels": slot_labels,
            "intent_labels": intent_labels,
        }
        #print(f"Batch ready: input_ids shape {batch['input_ids'].shape}")
        return batch
