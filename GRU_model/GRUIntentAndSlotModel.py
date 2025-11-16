import torch
import torch.nn as nn
import fasttext
import os 
from IntentAndSlotDataset import *
from fastText_training import get_vocab,create_intent2ind_slot2int
import math

class GRUEncoderFastText(nn.Module):

    """
    Modèle de test utilisant la classes de Dataset NLUIntentDatasetFastText
    Utilisant un RNN de type GRU. Les RNN sont conseillés pour traiter des 
    séquences, conservent l'ordre. 
    La couche d'embedding charge l'input matrix du modèle fasttext. 
    WORK IN PROGRESS
    """
    @staticmethod
    def load_embedding_weights(model,vocab:dict[str,int]):
        vec_size = model.get_dimension()
        embeddings = np.zeros((len(vocab)+1,vec_size)).astype(np.float32) # put the dimensions of the embedding matrix
        for word,ind in vocab.items():
            embeddings[ind] = model.get_word_vector(word)
        return embeddings


    def __init__(self,fastext_model_path:str,vocab:dict[str,int],hidden_dim, dropout,*args, **kwargs) -> None:
        """
        Constructeur du modèle
        Args:
            fastext_model_path (str): path du fichier modèle (.bin) généré par fasttext_model_construction
            padding_idx (int): l'index ou id du token utilisé pour faire le padding (remplissage)
            hidden_dim (_type_): dimension de la couche cachée du GRU
            dropout (_type_): _description_
        """
        super().__init__(*args, **kwargs)
        model = fasttext.load_model(fastext_model_path)
        self.embed_size = model.get_dimension()
        embed_weights = self.load_embedding_weights(model,vocab)
        weights_t = torch.from_numpy(embed_weights)
        self.embedding = nn.Embedding.from_pretrained(weights_t,freeze=False,padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)
        model = None
        
        # le dropout est utiles avec plusieurs GRU empilés 
        self.gru = nn.GRU(input_size=self.embed_size,hidden_size=hidden_dim,num_layers=1,batch_first=True,dropout=dropout,bidirectional=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        """

        Args:
            x (torch.Tensor): [N,L] N: batch size, L : sequence len (padded utt len)

        Returns:
            torch.Tensor: _description_
        """

        embed = self.dropout(self.embedding(x)) # [N,L] -> [ N, L, Embed_dim]
        output,hidden = self.gru(embed) # [N,L,Embed_dim] -> [N,L,2*Hidden] [2,N,Hidden]
        return output,hidden


class GRUSlotClassification(nn.Module):
    """
    Le modèle final doit etre adaptable à deux tâches de classification
    distinctes : intent classification et slot identification qui une
    classification des tokens.
    Cette couche est donc destinée à la tache de classification des slots
    Args:
        nn (_type_): _description_
    """
    def __init__(self,gru_hidden, nslots,dropout):
        """
        Args:
            gru_hidden (_type_): dimension de la couche cachée GRU bi-directionnel
            nclasses (_type_): nombre de classe/type de slots(60 sur le dataset restreint 
            aux langues latines)
        """
        super().__init__()
        self.decoder = nn.Linear(2*gru_hidden, nslots)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)


    def forward(self,x): # [N,Seq,2*Hidden]
        output = self.dropout(self.relu(self.decoder(x)))
        return output  # [N,Seq,nslots]


class GRUIntentClsModel(nn.Module):
    """
    Le modèle presque complet GRU + Tete de classification
    Args:
        nn (_type_): _description_
    """
    def __init__(self,gru_hidden_dim:int,nbintents,dropout=0.1):
        """
        Args:
            fastext_model_path (str): path du modele fasttext (.bin)
            padding_idx (int): index dans l'input matrix de la valeur de padding (0)
            seq_len (int): longueur max des utt
            hidden_dim (int): dimension de la couche cachée GRU
            nclasses (_type_): nombre de classes d'intent
            dropout (float, optional): _description_. Defaults to 0.5.
        """
        super().__init__()
        self.decoder = nn.Linear(2*gru_hidden_dim, nbintents)
        self.dropout = nn.Dropout(p=dropout)
        #self.logsft = nn.LogSoftmax(dim=1)
        self.init_weights()
    
    #il est important de mettre des poids car on a une répartition très innégalitaire des données
    #et que si une classe est surreprésentée, il y a un risque de surraprentissage.
    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):  # [N,Seq,2*hidden]

        x_last = x[:,-1,:] # [N,2*Hidden]
        x_out = self.dropout(self.decoder(x_last)) # [N,nbintent]
        #output =  self.logsft(x_out)
        return x_out  # [N,nbintent]


class GRUModel(nn.Module):
    """
    Cette classe nous permet d'utiliser nos fonctions précédentes pour créer notre modèle GRU.
    """
    def __init__(self,fastext_model_path:str,
                 hidden_dim:int,nbintents,nbslots,vocab:dict[str,int],dropout=0.1, *args, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_encoder = GRUEncoderFastText(fastext_model_path,vocab,hidden_dim,dropout)
        self.intent_cls = GRUIntentClsModel(hidden_dim,nbintents,dropout)
        self.slots_cls = GRUSlotClassification(hidden_dim,nbslots,dropout)

    def forward(self, src): # [N,Seq] 
        # base model
        enc_out,enc_hidden =  self.gru_encoder(src) # [N,Seq,2*hidden] * math.sqrt(self.hidden_dim)
        # classifier model
        intent_logits =  self.intent_cls(enc_out) # [N,nbintent]
        slot_logits = self.slots_cls(enc_out) # [N,Seq,nbslots]
        return intent_logits,slot_logits  # logit 


if __name__=='__main__':


    fst250bin =  os.path.join(os.path.dirname(os.path.realpath(__file__)),"models","fst250_10K_all.model")
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

    gru_model = GRUModel(fst250bin,250,nb_intent,nb_slots,vocab,nb_intent=nb_intent)

    out_intent,out_slots = gru_model(uut_0)
    print(out_intent.size())
    print(out_slots.size())
    _, intent_pred = torch.max(out_intent, 1)
    print(intent_pred)



