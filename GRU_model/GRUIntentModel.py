import torch
import torch.nn as nn
import fasttext
import os 
from dataset_extract import *
from fastText_training import get_vocab

class GRUBaseFastText(nn.Module):

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


    def __init__(self,fastext_model_path:str,vocab:dict[str,int],padding_idx:int ,hidden_dim, dropout,*args, **kwargs) -> None:
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
        self.embedding = nn.Embedding.from_pretrained(weights_t,freeze=False)
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
        output = self.dropout(output)
        return output,hidden


class GRUIntentClassification(nn.Module):
    """
    Le modèle final doit etre adaptable à deux tâches de classification
    distinctes : intent classification et slot identification qui une
    classification des tokens.
    Cette couche est donc destinée à la tache de classification des intents
    Args:
        nn (_type_): _description_
    """
    def __init__(self,gru_hidden, nclasses,dropout):
        """
        Args:
            gru_hidden (_type_): dimension de la couche cachée GRU bi-directionnel
            nclasses (_type_): nombre de classe d'intent (60 sur le dataset restreint 
            aux langues latines)
        """
        super().__init__()
        self.decoder = nn.Linear(2*gru_hidden, nclasses)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)


    def forward(self,x):
        output = self.dropout(self.relu(self.decoder(x)))
        return output


class GRUIntentClsModel(nn.Module):
    """
    Le modèle presque complet GRU + Tete de classification
    Args:
        nn (_type_): _description_
    """
    def __init__(self, fastext_model_path:str,padding_idx:int,
                 hidden_dim:int,nclasses,vocab:dict[str,int],dropout=0.1):
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
        self.base = GRUBaseFastText(fastext_model_path,vocab ,padding_idx, hidden_dim, dropout)
        self.classifier = GRUIntentClassification(hidden_dim, nclasses,dropout)

    def forward(self, x):
        # base model
        x_out,x_hidden =  self.base(x) # hidden : [2,N,Hidden], x_out : [N,L,2*H ]
        #x_hidden_p = torch.permute(x_hidden,(1,0,2))
        x_out_last = x_out[:,-1,:] # [N,2*Hidden]
        #x_out_flat = torch.flatten(x_hidden_p,1,2) # [ N,2*Hidden]
        # classifier model
        output =  self.classifier(x_out_last) # fill me 
        return output



if __name__=='__main__':


    fst250bin =  os.path.join(os.path.dirname(os.path.realpath(__file__)),"models","fst250_10K_all.model")
    model = fasttext.load_model(fst250bin)
    corpus_path =  join(os.path.dirname(os.path.realpath(__file__)),"corpuses","extract0_corpus.txt")

    vocab = get_vocab(corpus_path)

    ds_path = join(os.path.dirname(os.path.realpath(__file__)),"massive_extract")
    files_list = ["de-DE.jsonl","en-US.jsonl","es-ES.jsonl","fr-FR.jsonl","it-IT.jsonl"]

    all_dataset_files = [join(ds_path, f) for f in files_list ]
    

    models_dir = join(os.path.dirname(os.path.realpath(__file__)),"models")
    #fastext_model_path = join(models_dir, "fst250.bin")

    my_loader_fast,seq_len = get_loader_fasttext(all_dataset_files,True,fst250bin,vocab,32,221)

    model = GRUIntentClsModel(fst250bin,padding_idx=0,hidden_dim=32,nclasses=60,vocab=vocab,dropout=0.1)

    #fasttextDS = NLUIntentDatasetFastText(all_dataset_files,True,fastext_model_path)
    #second_tensor = fasttextDS[1]
    my_iter_fast = iter(my_loader_fast)
    batch_0,target_0 = next(my_iter_fast)
    print(batch_0.size())

    out = model(batch_0)
    print(out.size())



