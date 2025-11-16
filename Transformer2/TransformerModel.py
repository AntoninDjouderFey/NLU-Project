import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import fasttext
import os 
from dataset_extract import *
from fastText_training import get_vocab
from IntentAndSlotDataset import *

#Cette classe aura pour role de transformer une phrase en une représentation vectorielle contextualisée
class TransformerEncoder(nn.Module):

    @staticmethod
    #Charge les vecteurs fastText et créé ensuite une matrice d'embeddings
    def load_embedding_weights(model,vocab:dict[str,int]):
        vec_size = model.get_dimension()
        embeddings = np.zeros((len(vocab)+1,vec_size)).astype(np.float32) # put the dimensions of the embedding matrix
        for word,ind in vocab.items():
            embeddings[ind] = model.get_word_vector(word)
        return embeddings
    
    def __init__(self,fastext_model_path:str ,vocab:dict[str,int], nhead, nhid, nlayers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        '''
        ntoken: the size of vocabulary
        nhid: the hidden dimension of the model.
        We assume that embedding_dim = nhid
        nlayers: the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead: the number of heads in the multiheadattention models
        dropout: the dropout value
         '''
        self.model_type = "Transformer"
        #Charge FastText, crée l’Embedding PyTorch à partir des vecteurs pré-entraînés
        model = fasttext.load_model(fastext_model_path)
        self.embed_size = model.get_dimension()
        embed_weights = self.load_embedding_weights(model,vocab)
        weights_t = torch.from_numpy(embed_weights)
        self.encoder = nn.Embedding.from_pretrained(weights_t,freeze=False,padding_idx=0)
        #génére un politionnal encoding pour garder l'ordre des mots
        #car avec des transformers les mots sont traités parallèlement et non séquentiellement
        self.pos_encoder = PositionalEncoding(nhid, dropout) #fill me, the PositionalEncoding class is implemented in the next cell
        encoder_layers =  nn.TransformerEncoderLayer(nhid, nhead, nhid, dropout,batch_first=True) #fill me we assume nhid = d_model = dim_feedforward # 1 - the input dim, 2- number of heads, 3- dim_feeforward, 4- dropout
        self.transformer_encoder =  nn.TransformerEncoder(encoder_layers, nlayers,enable_nested_tensor=False) #fill me, we want nlayers of encoder_layers
        self.nhid = nhid #dimension des embeddings transformés
        self.init_weights()

    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.nhid) 
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask) 
        return output
    

class IntentClassificationHead(nn.Module):
    """
    Intent classification head.
    The output is [N,nb_intent] where last dimension are intent
    loggits. For CrossEntropyLoss

    Args:
        nn (_type_): _description_
    """
    def __init__(self,nhid ,nclasses,dropout=0.1):
        super(IntentClassificationHead, self).__init__()
        self.decoder = nn.Linear(nhid, nclasses)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, enc_out):
        output = self.dropout(self.decoder(enc_out)) # [N,seq,nhid]
        return output[:,-1,:] # last vector of the sequence [N,nb_intent]
    

class SlotClassificationHead(nn.Module):
    """
    Slot classification head, output: [N,Seq,nb_slots] 
    last  dimension contains slots loggits ( non normalized probabilities
    of slot )

    Args:
        nn (_type_): _description_
    """
    def __init__(self,nhid ,nslots,dropout=0.1):
        super(SlotClassificationHead, self).__init__()
        self.decoder = nn.Linear(nhid, nslots)
        self.dropout = nn.Dropout(p=dropout)
        self.logsoftmax = nn.LogSoftmax(dim=2)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, enc_out):
        output = self.dropout(self.decoder(enc_out)) # [N,seq,nslots]
        loggit_out = self.logsoftmax(output)
        return loggit_out
    
class Model(nn.Module):
    """
    The full model, Transformer encoder --> tuple intent_cls , slot_cls

    Args:
        nn (_type_): _description_
    """
    def __init__(self, fastext_model_path,vocab,nb_intent,nb_slots,nhead, nhid, nlayers, dropout=0.1):
        super(Model, self).__init__()
        self.encoder = TransformerEncoder(fastext_model_path,vocab, nhead, nhid, nlayers, dropout)
        self.intent_cls = IntentClassificationHead(nhid, nb_intent,dropout)
        self.slots_cls = SlotClassificationHead(nhid,nb_slots,dropout)

    def forward(self, src, src_mask):
        # base model
        enc_out =  self.encoder(src, src_mask)
        # classifier model
        intent_logits =  self.intent_cls(enc_out)
        slot_logits = self.slots_cls(enc_out)
        return intent_logits,slot_logits
    

class PositionalEncoding(nn.Module):
    """
    Position encoding adds sequencing information to embedded sequence

    Args:
        nn (_type_): _description_
    """
    def __init__(self, nhid, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, nhid)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, nhid, 2).float() * (-math.log(10000.0) / nhid)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
    


if __name__ == "__main__":
    fst250bin =  os.path.join(os.path.dirname(os.path.realpath(__file__)),"models","fst250_10K_all.model")
    model = fasttext.load_model(fst250bin)
    corpus_path =  join(os.path.dirname(os.path.realpath(__file__)),"corpuses","extract0_corpus.txt")

    vocab = get_vocab(corpus_path)

    train_dataset_path =  join(os.path.dirname(os.path.realpath(__file__)),"datasets","train_formatted_dataset.txt")
    test_dataset_path =  join(os.path.dirname(os.path.realpath(__file__)),"datasets","test_formatted_dataset.txt")

    intent2id,slot2id= create_intent2ind_slot2int([train_dataset_path,test_dataset_path])

    data_loader,nb_intent,nb_slots = get_loader_IntentAndSlot(train_dataset_path,vocab,intent2id,slot2id,32)
    my_iter_fast = iter(data_loader)
    uut_0,intent_0,slot_0 = next(my_iter_fast)
    print(uut_0.size())
    print(f"slot_0 {slot_0.size()}")

    encoder_model = Model(fst250bin,vocab,nb_intent=nb_intent,
                          nb_slots=nb_slots,nhead=2,nhid=250,nlayers=2)


    intent_logits,slot_logits = encoder_model(uut_0,None)
    print(f"intent output {intent_logits.size()}")
    print(f"slot output {slot_logits.size()}")

