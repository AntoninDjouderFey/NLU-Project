import json
import torch
from torch.utils.data import DataLoader, Dataset
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from os import listdir
import os
from os.path import isfile, join
import numpy as np
import unicodedata
import linecache
import fasttext
from fastText_training import get_vocab,create_intent2ind_slot2int,compute_class_weights


class IntentAndSlotDS(Dataset):
    """
    Dataset prenant en entrée un fichier txt au format :
    utt|intent|slots generéré par create_preproc_dataset fastText_training
    """
    



    def __init__(self,dataset_path:str,vocab:dict[str,int],intent2id:dict[str,int],slot2id:dict[str,int]):
        super().__init__()

        self.vocab = vocab
        self.vocab_size = len(vocab)
        # L'entrée du token_id = 0 correspond à </s> dans un modèle
        # fasttext je l'ai utilisé pour le padding on peut faire mieux.
        self.padding_idx = 0
        self.intent2ind = intent2id
        self.slot2ind = slot2id
        self.slot2ind['None'] = self.padding_idx
        self.utt_max_len = 0
        self.length = 0
        self.ds_path = dataset_path
        with open(dataset_path, "rb") as f:
            self.length = sum(1 for _ in f)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Utilise les token_id du modèle fasttext.
        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        line = linecache.getline(self.ds_path,idx).strip()
        if line=='' and idx +1 < self.length:
            line = linecache.getline(self.ds_path,idx+1).strip()
        utt,intent,slots = line.split('|')

        utt_sequence = [ self.vocab[word.strip()] for word in utt.split(' ') if len(word.strip())>0 ]
        slot_sequence = [ self.slot2ind[word.strip()] for word in slots.split(' ') if len(word.strip())>0]

        if len(utt_sequence) != len(slot_sequence):
            raise ValueError(f"invalid entry {utt}, {slots} in {self.ds_path} at {idx}") 


        intent_ind = [self.intent2ind[intent]]

        sample = {
            "utt_sequence": torch.tensor(utt_sequence),
            "intent": torch.tensor(intent_ind),
            "slot_sequence": torch.tensor(slot_sequence)
        }
        return sample
    
def get_loader_IntentAndSlot(
  data_path:str,      
  vocab:dict[str,int],
  intent2id:dict[str,int],
  slot2id:dict[str,int],
  batch_size:int=32,
  seq_len:int=-1
):
    
    """
    Construit des batch de tuple  ([N,Seq],[N],[N,Seq]) (utt,intent,annot_utt)  N: batch Size , Seq nombre max de mots de l'utt
    """
    dataset = IntentAndSlotDS( #Il s'agit de l'itérateur précédemment défini
        data_path,
        vocab=vocab,intent2id=intent2id,slot2id=slot2id)
    
    #Un collator à préparer et regrouper nos données en un seul batch afin qu'ils soit formatés pour être traités par un modèle
    def MyCollatorFasttext(batch):
        #on utilise le padding pour matcher la longueur des séquences dans le même batch 
        if seq_len > 0:
            utt_0 = batch[0]["utt_sequence"]
            #ce padding à utt_max_len assure des batch ayant tous la meme dimension [ N, utt_max_len]
            padded_batch0 = torch.nn.ConstantPad1d((0, seq_len - utt_0.size(0)), dataset.padding_idx)(utt_0)
            batch[0]["source_sequence"] = padded_batch0
        utt_sequences = pad_sequence(
            [sample["utt_sequence"] for sample in batch], batch_first=True,
              padding_value= dataset.padding_idx )
        slot_sequence = pad_sequence(
            [sample["slot_sequence"] for sample in batch], batch_first=True,
              padding_value= dataset.padding_idx )
        intent =torch.stack([sample["intent"] for sample in batch],dim=0).squeeze()
        return utt_sequences, intent,slot_sequence # data[0] : doc, data[1]: label
    
    data_loader = DataLoader( #Le data_loader est une classe pytorch qui sert à construire des batchs
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=MyCollatorFasttext,
        pin_memory=True,
        drop_last=True,
    )
    return data_loader,len(dataset.intent2ind),len(dataset.slot2ind)


if __name__ == "__main__":

    train_dataset_path =  join(os.path.dirname(os.path.realpath(__file__)),"datasets","train_formatted_dataset.txt")
    test_dataset_path =  join(os.path.dirname(os.path.realpath(__file__)),"datasets","test_formatted_dataset.txt")
    corpus_path =  join(os.path.dirname(os.path.realpath(__file__)),"corpuses","extract0_corpus.txt")
    model_path = join(os.path.dirname(os.path.realpath(__file__)),"models","fst250_10K_all.model")
    model = fasttext.load_model(model_path)
        
    #le vocabulaire réel construit par fasttext avec les sous-mots
    word2inds = get_vocab(corpus_path)

    intent2id,slot2id= create_intent2ind_slot2int([train_dataset_path,test_dataset_path])

    intent_weights,slot_weights = compute_class_weights(intent2id,slot2id,[train_dataset_path,test_dataset_path])
    print(f"intent weight {intent_weights}")
    print(f"slot weight {slot_weights}")
    
    my_dataset = IntentAndSlotDS(train_dataset_path,word2inds,intent2id,slot2id)

    second_tensor = my_dataset[1]
    print(second_tensor)

    data_loader,nb_intents,nb_slots = get_loader_IntentAndSlot(train_dataset_path,word2inds,intent2id,slot2id,32)
    print(f"nb_intent:{nb_intents}, nb_slots:{nb_slots}")
    my_iter_fast = iter(data_loader)
    uut_0,intent_0,slot_0 = next(my_iter_fast)
    print(uut_0.size())
    print(intent_0.size())
    print(slot_0.size())
