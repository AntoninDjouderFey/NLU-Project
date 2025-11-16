import torch
import torch.nn.functional as F
import os
from GRUIntentAndSlotModel import GRUModel
from IntentAndSlotDataset import *
from torch import optim
from tqdm import tqdm
import time
import copy
from fastText_training import get_vocab
import fasttext
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import json

#données de score d'entrainements qui seront plottées
history = {
    "intent_train": [],
    "slot_train": [],
    "intent_val": [],
    "slot_val": [],
    "precision_intent": [],
    "recall_intent": [],
    "fbeta_intent": [],
    "precision_slot": [],
    "recall_slot": [],
    "fbeta_slot": []
}

"""
This script trains Transformer model with one optimzer
Two losses : one for 


"""

def get_parameters_count(model):
   """
   Donne la taille du modele
   """
   paramerters_count = 0
   for el1, el2 in model.named_parameters():

    if len(el2.shape) == 2:
        paramerters_count += el2.shape[0] * el2.shape[1]
    else:
        paramerters_count += el2.shape[0]

    return paramerters_count

def get_device():
    """
    Cherche si une GPU est dispo
    """
    if torch.backends.mps.is_available():
        print("Using mps device")
        device = torch.device("mps")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return device

def checkpoint(model, filename):
    """
    Sauvegarde un modèle
    """
    torch.save(copy.deepcopy(model.state_dict()), filename)

def plotter(history:dict[str,float]):
    f, (ax1, ax2) = plt.subplots(1,2,figsize=(15,9))
    ax1.plot(history["intent_train"], label="Intent Train Loss")
    ax1.plot(history["intent_val"], label="Intent Val Loss")
    ax1.plot(history["slot_train"], label="Slot Train Loss")
    ax1.plot(history["slot_val"], label="Slot Val Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("Évolution des pertes d'entraînement et validation")
    ax1.legend()
    ax2.plot(history["recall_intent"], label="Recall Intent")
    ax2.plot(history["precision_intent"], label="Precision Intent")
    ax2.plot(history["fbeta_intent"], label="Fbeta Intent")
    ax2.plot(history["recall_slot"], label="Recall Slot")
    ax2.plot(history["precision_slot"], label="Precision Slot")
    ax2.plot(history["fbeta_slot"], label="Fbeta Slot")
    ax2.set_title("Evolution des Fbeta, Précisions et Rappels d'entrainements et de validations")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Stats")
    ax2.legend()
    plt.grid(True)
    plt.savefig("GRU_training_data.png", dpi=150)
    plt.close()

def resume_model(model, filename):
        """
        Charge un modèle à partir d'un ficher state_dict
        """
        loaded = torch.load(filename)
        model.load_state_dict(loaded)

if __name__ == "__main__":
   
    device=get_device()
    print(f"train using device {device}")

    #fst100bin =  os.path.join(os.path.dirname(os.path.realpath(__file__)),"models","fst100_latin.model")
    fasttext_path =  os.path.join(os.path.dirname(os.path.realpath(__file__)),"models","fst100_all.model")

    model = fasttext.load_model(fasttext_path)
    corpus_path =  os.path.join(os.path.dirname(os.path.realpath(__file__)),"corpuses","extract0_corpus.txt")

    vocab = get_vocab(corpus_path)
    print(f"Vocab size {len(vocab)}")
    model = None

    ds_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"massive_extract")
    files_list = ["de-DE.jsonl","en-US.jsonl","es-ES.jsonl","fr-FR.jsonl","it-IT.jsonl"]
    #files_list = ["en-US.jsonl","fr-FR.jsonl"]

    train_dataset_path =  join(os.path.dirname(os.path.realpath(__file__)),"datasets","train_formatted_dataset.txt")
    test_dataset_path =  join(os.path.dirname(os.path.realpath(__file__)),"datasets","test_formatted_dataset.txt")

    train_nofr_dataset_path =  join(os.path.dirname(os.path.realpath(__file__)),"datasets","train_nofr_formatted_dataset.txt")
    test_nofr_dataset_path =  join(os.path.dirname(os.path.realpath(__file__)),"datasets","test_nofr_formatted_dataset.txt")
    fr_eval_dataset_path = join(os.path.dirname(os.path.realpath(__file__)),"datasets","fr_eval_dataset.txt")

    intent2id,slot2id= create_intent2ind_slot2int([train_dataset_path,test_dataset_path])

    intent_weights,slot_weights = compute_class_weights(intent2id,slot2id,[train_dataset_path,test_dataset_path])
    intent_weights_t = torch.FloatTensor(intent_weights).to(device)
    slot_weights_t = torch.FloatTensor(slot_weights).to(device)
    # dict used to compute sample weights
    intent_weights_dict = { id:intent_weights[id] for id in range(len(intent_weights)) }
    slot_weights_dict = { id:slot_weights[id] for id in range(len(slot_weights)) }


    batch_size=32

    all_dataset_files = [os.path.join(ds_path, f) for f in files_list ]
    
    val_loader,nb_intent_test,nb_slots_test = get_loader_IntentAndSlot(fr_eval_dataset_path,vocab,intent2id,slot2id,batch_size)
    print(f"test intent:{nb_intent_test} slots:{nb_slots_test}")

    nb_slots = len(slot2id)
    nb_intent = len(intent2id)

    model = GRUModel(fasttext_path,hidden_dim=250,nbintents=nb_intent,nbslots=nb_slots,vocab=vocab,dropout=0.1)


    model_name = "GRU_IntentSlots_model_nofr.pt"

    nb_params = get_parameters_count(model)
    print(f"model size {nb_params}")

    

    model_path =  join(os.path.dirname(os.path.realpath(__file__)),model_name)
    if os.path.isfile(model_path):
        print(f"Resuming model {model_path}")
        resume_model(model,model_path)
    else:
        raise Exception(f"Model {model_name} not found")
    
    model.to(device)

    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_f_intent = torch.nn.CrossEntropyLoss(ignore_index=0,weight=intent_weights_t)
    loss_f_slot = torch.nn.CrossEntropyLoss(weight=slot_weights_t)

    train_loss = 0
    val_loss = 0
    correct = 0
    count = 0

    epochs = 1

        #y_test_t = y_test_t.to(device)

    best_val_loss = -1
    best_sum_fbeta = -1
    intent_best_val_loss = -1


    # inspired from https://github.com/ray075hl/Bi-Model-Intent-And-Slot/blob/master/train.py


    for epoch in range(epochs):
        t = time.time()
         
        model.eval()
        count_val = 0
        intent_val_loss = 0
        slot_val_loss = 0
        sum_val_loss = 0

        val_intent_pred = []
        val_intent_true = []

        val_slot_pred = []
        val_slot_true = []

        for utt_x,intent_y,slot_y in tqdm(val_loader) : # intent_y, slot_y ground truth 
            utt_x = utt_x.to(device)
            intent_y = intent_y.to(device)
            slot_y = slot_y.to(device)

            intent_out,slot_out = model(utt_x) # [N,nbIntent] , [N,Seq,nbSlots]
            intent_loss = loss_f_intent(intent_out,intent_y)
            intent_val_loss += intent_loss.item() * intent_out.size(0)

            count_val += intent_out.size(0)

            _, intent_pred = torch.max(intent_out, 1) # index de la valeur max d'intent
            val_intent_pred.extend(intent_pred.cpu().numpy())
            val_intent_true.extend(intent_y.cpu().numpy())

            slot_out_p = torch.permute(slot_out,(0,2,1)) # [N,Seq,nbSlots] -> [N,nbSlots,Seq]
            slot_loss = loss_f_slot(slot_out_p,slot_y)
            slot_val_loss += slot_loss.item()
            sum_val_loss = intent_val_loss + slot_val_loss

            _, slot_pred = torch.max(slot_out,dim=2)# [N,Seq,nbSlots] -> [N,Seq]
            slof_pred_flat = torch.flatten(slot_pred) # [N,Seq] -> [ N*Seq]
            slot_y_flat = torch.flatten(slot_y)
            val_slot_pred.extend(slof_pred_flat.cpu().numpy())
            val_slot_true.extend(slot_y_flat.cpu().numpy())



        intent_sample_weights = compute_sample_weight(class_weight=intent_weights_dict,y=val_intent_true)
        precision,recall,fbeta,_ = precision_recall_fscore_support(val_intent_true,val_intent_pred,average='weighted',sample_weight=intent_sample_weights)

        slot_sample_weights = compute_sample_weight(class_weight=slot_weights_dict,y=val_slot_true)
        precision_slot,recall_slot,fbeta_slot,_ = precision_recall_fscore_support(val_slot_true,val_slot_pred,average='weighted')



        print('epoch: {:04d}\n'.format(epoch+1),
                'intent_val_loss: {:.4f}\n'.format(intent_val_loss/count_val),
                'slot_val_loss: {:.4f}\n'.format(slot_val_loss/count_val),
                'time: {:.4f}s'.format(time.time() - t))
        
        print('intent precision: {:.4f}\n'.format(precision),'intent recall: {:.4f}'.format(recall),
                'intent fbeta: {:.4f}\n'.format(fbeta))
        
        print('slot precision: {:.4f}\n'.format(precision_slot),'slot recall: {:.4f}'.format(recall_slot),
                'slot fbeta: {:.4f}\n'.format(fbeta_slot))
        
    print("--------------------------------------")

"""
before :
 intent_val_loss: 2.4071
 slot_val_loss: 0.0775
 time: 4.5649s
intent precision: 0.6574
 intent recall: 0.6023 intent fbeta: 0.5866

slot precision: 0.9082
 slot recall: 0.8798 slot fbeta: 0.8894

 after:
intent_val_loss: 1.9538
 slot_val_loss: 0.0686
 time: 4.5226s
intent precision: 0.6865
 intent recall: 0.6529 intent fbeta: 0.6437

slot precision: 0.9226
 slot recall: 0.8654 slot fbeta: 0.8859
"""