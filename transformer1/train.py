#!/usr/bin/env python3
# coding: utf-8
"""
train_joint.py
Train a joint intent + slot model.

python train_joint.py --slots_csv csv/slots_all_train.csv --slots_valid_csv csv/slots_all_valid.csv --slot_labels_json csv/slot_labels.json --model_name xlm-roberta-base  --output_dir models --epochs 5 --batch_size 8 --lr 2e-5
"""
from tqdm import tqdm  

from torch.nn.utils.rnn import pad_sequence
import os
import json
import random
import argparse
from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
import torch_directml
#I don't want to add 10 line of import and this work
from transformers import AutoTokenizer, AutoConfig, AutoModel, get_linear_schedule_with_warmup

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from seqeval.metrics import f1_score as seq_f1, precision_score as seq_precision, recall_score as seq_recall
#so I know where it's from
from data_loader import NLUdataset, DataCollatorForNLU

# ---------------------------
# Joint model definition
# ---------------------------
class JointIntentSlotModel(nn.Module):
    def __init__(self, pretrained_name: str, num_intents: int, num_slot_labels: int, dropout=0.1):
        super().__init__()
        config = AutoConfig.from_pretrained(pretrained_name)
        self.encoder = AutoModel.from_pretrained(pretrained_name, config=config)
        hidden_size = config.hidden_size

        self.intent_dropout = nn.Dropout(dropout)
        self.intent_classifier = nn.Linear(hidden_size, num_intents)

        self.slot_dropout = nn.Dropout(dropout)
        self.slot_classifier = nn.Linear(hidden_size, num_slot_labels)

        self.intent_loss_fct = nn.CrossEntropyLoss()
        self.slot_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids=None, attention_mask=None, intent_labels=None, slot_labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        sequence_output = outputs.last_hidden_state    # (B, L, H)

        # pooled: utilise le premier token (ça marche pour BERT et XLM-R)
        pooled = sequence_output[:, 0, :]

        intent_logits = self.intent_classifier(self.intent_dropout(pooled))         # (B, num_intents)
        slot_logits = self.slot_classifier(self.slot_dropout(sequence_output))      # (B, L, num_slot_labels)

        loss = None
        if intent_labels is not None and slot_labels is not None:
            intent_loss = self.intent_loss_fct(intent_logits.view(-1, intent_logits.size(-1)), intent_labels.view(-1))
            slot_loss = self.slot_loss_fct(slot_logits.view(-1, slot_logits.size(-1)), slot_labels.view(-1))
            loss = intent_loss + slot_loss

        return {"loss": loss, "intent_logits": intent_logits, "slot_logits": slot_logits}

# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def decode_slots(slot_logits: np.ndarray, slot_label_list: List[str], ignore_label_id: int = -100) -> List[List[str]]:
    """
    Convert slot logits (B, L, C) -> list of BIO tag sequences as strings (one list per example),
    ignoring positions where label id would be ignore_label_id.
    """
    preds = np.argmax(slot_logits, axis=-1)
    # preds is (B, L)
    results = []
    for seq in preds:
        tags = [slot_label_list[p] if p >= 0 and p < len(slot_label_list) else "O" for p in seq]
        results.append(tags)
    return results

def align_preds_and_labels(pred_tags: List[List[str]], label_ids: torch.Tensor, slot_label_list: List[str], ignore_idx: int = -100):
    id2lab = {i: l for i, l in enumerate(slot_label_list)}
    true = []
    preds = []
    label_np = label_ids.cpu().numpy()
    for p_seq, g_seq in zip(pred_tags, label_np):
        out_p = []
        out_g = []
        for p_tag, g_id in zip(p_seq, g_seq):
            if g_id == ignore_idx:
                continue
            out_p.append(p_tag)
            out_g.append(id2lab.get(int(g_id), "O"))
        preds.append(out_p)
        true.append(out_g)
    return preds, true

# ---------------------------
# Training / Eval functions
# ---------------------------
def evaluate(model, loader, device, slot_label_list):
    model.eval()
    all_intent_preds, all_intent_true = [], []
    all_slot_preds_tags = []
    all_slot_true_ids = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", unit="batch"):
            input_ids = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            slot_labels = batch['slot_labels'].to(device)
            intent_labels = batch['intent_labels'].to(device)

            out = model(input_ids=input_ids, attention_mask=attn, intent_labels=None, slot_labels=None)
            intent_logits = out['intent_logits'].detach().cpu().numpy()
            slot_logits = out['slot_logits'].detach().cpu().numpy()

            # intents
            intent_pred = np.argmax(intent_logits, axis=-1)
            all_intent_preds.extend(intent_pred.tolist())
            all_intent_true.extend(intent_labels.detach().cpu().numpy().tolist())

            # slot
            slot_pred_tags = decode_slots(slot_logits, slot_label_list)
            all_slot_preds_tags.extend(slot_pred_tags)


            batch_size = slot_labels.size(0)
            for i in range(batch_size):
                mask = attn[i].bool()
                true_seq = slot_labels[i][mask].detach().cpu()
                all_slot_true_ids.append(true_seq)

    # stuff000000
    if len(all_slot_true_ids) > 0:
        padded_true = pad_sequence(all_slot_true_ids, batch_first=True, padding_value=-100)
    else:
        padded_true = torch.empty((0, 0), dtype=torch.long)

    # aligner les prédictions et les étiquettes, en supprimant les -100
    per_sample_true_ids = []
    N = padded_true.shape[0]
    for i in range(N):
        valid_mask = (padded_true[i] != -100)
        per_sample_true_ids.append(padded_true[i][valid_mask].tolist()) 

    pred_final = []
    true_final = [] 
    id2lab = {i: l for i, l in enumerate(slot_label_list)}
    for pred_tags, true_ids in zip(all_slot_preds_tags, per_sample_true_ids):
        L = len(true_ids)
        pred_trim = pred_tags[:L]
        true_labels = [id2lab[t] for t in true_ids] 
        pred_final.append(pred_trim)
        true_final.append(true_labels)

    # métriques de slot
    slot_precision = seq_precision(true_final, pred_final) if len(true_final) > 0 else 0.0
    slot_recall = seq_recall(true_final, pred_final) if len(true_final) > 0 else 0.0
    slot_f1 = seq_f1(true_final, pred_final) if len(true_final) > 0 else 0.0

    # métriques d'intent
    intent_acc = accuracy_score(all_intent_true, all_intent_preds) if len(all_intent_true) > 0 else 0.0
    intent_f1 = f1_score(all_intent_true, all_intent_preds, average='weighted', zero_division=0) if len(all_intent_true) > 0 else 0.0

    return {
        "intent_acc": intent_acc,
        "intent_f1": intent_f1,
        "slot_precision": slot_precision,
        "slot_recall": slot_recall,
        "slot_f1": slot_f1
    }
# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots_train_csv", type=str, required=True)
    parser.add_argument("--slots_valid_csv", type=str, required=True)
    parser.add_argument("--slot_labels_json", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        if torch.backends.mps.is_available():  
            device = torch.device("mps")
        elif torch.cuda.is_available():       
            device = torch.device("cuda")
        elif hasattr(torch_directml, "device"): 
            device = torch_directml.device()
        else:
            device = torch.device("cpu")

    print(f"[INFO] Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # load slot labels
    with open(args.slot_labels_json, "r", encoding="utf-8") as fh:
        slot_label_list = json.load(fh)["labels"]

    # datasets
    train_ds = NLUdataset(args.slots_train_csv, tokenizer)
    valid_ds = NLUdataset(args.slots_valid_csv, tokenizer,
                               slot_label_list=train_ds.slot_label_list,
                               intent_label_list=train_ds.intent_label_list)

    collator = DataCollatorForNLU(tokenizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    num_intents = len(train_ds.intent_label_list)
    num_slot_labels = len(train_ds.slot_label_list)
    print(f"[INFO] num_intents={num_intents} num_slot_labels={num_slot_labels}")

    model = JointIntentSlotModel(args.model_name, num_intents, num_slot_labels)
    model.to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    #J'ai essayer plusieurs optimisers, adamW n'est pas le plus rapide(parce que j'utilise direct ML) mais j'ai fais un modèle complet avec donc je le laisse là
    #Il n'est pas trop lent non plus par apport aux autres
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.0*total_steps), num_training_steps=total_steps)


# ---------------------------
# Training loop with progress bar
# ---------------------------
    best_val = -1.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        # On utilise tqdm pour avoir un jolie barre de progression (ça ne sert à rien dans le programme)
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            slot_labels = batch['slot_labels'].to(device)
            intent_labels = batch['intent_labels'].to(device)

            outputs = model(
                input_ids=input_ids, 
                attention_mask=attn, 
                intent_labels=intent_labels, 
                slot_labels=slot_labels
            )
            loss = outputs['loss']
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            # Update tqdm bar with current loss
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / max(1, len(train_loader))
        val_metrics = evaluate(model, valid_loader, device, train_ds.slot_label_list)

        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={avg_loss:.4f} | "
              f"val_intent_f1={val_metrics['intent_f1']:.4f} val_slot_f1={val_metrics['slot_f1']:.4f}")

        # choose primary metric as joint: intent_f1 * slot_f1
        joint_metric = val_metrics['intent_f1'] * val_metrics['slot_f1']
        if joint_metric > best_val:
            best_val = joint_metric
            save_path = os.path.join(args.output_dir, "best_model")
            os.makedirs(save_path, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "slot_label_list": train_ds.slot_label_list,
                "intent_label_list": train_ds.intent_label_list,
                "model_name": args.model_name
            }, os.path.join(save_path, "pytorch_model.bin"))
            tokenizer.save_pretrained(save_path)
            print(f"[INFO] New best joint metric={joint_metric:.6f} saved to {save_path}")

    # final eval
    final_val = evaluate(model, valid_loader, device, train_ds.slot_label_list)
    print("Final validation metrics:", final_val)
    # save final model
    final_path = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_path, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "slot_label_list": train_ds.slot_label_list,
        "intent_label_list": train_ds.intent_label_list,
        "model_name": args.model_name
    }, os.path.join(final_path, "pytorch_model.bin"))
    tokenizer.save_pretrained(final_path)
    print(f"[INFO] Final model saved to {final_path}")

if __name__ == "__main__":
    main()
