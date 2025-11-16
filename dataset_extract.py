import json
import torch
from torch.utils.data import DataLoader, Dataset
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from os import listdir
import os
from os.path import isfile, join
import fasttext
import numpy as np
import unicodedata
from fastText_training import get_vocab


#Comme son nom l'indique cette fonction extrait du json un tuple composé de l'intent et de la phrase
def extract_intent_fields(json_obj:dict)-> tuple[str,str]:
    if 'utt' not in json_obj.keys():
        raise ValueError("Invalid sample no UTT")
    if 'intent' not in json_obj.keys():
        raise ValueError("Invalid sample no intent")
    utt_norm = normalize(json_obj['utt'])
    return utt_norm,json_obj['intent']


def normalize(line):
    """Normalize a line of text and split into two at the tab character"""
    line = unicodedata.normalize("NFKC", line.strip())
    return line

#On créé un itérateur qui renvoie un tuple (token(utt),token(intent)) qui est produit à partir d'un ou plusieurs fichiers du dataset
#Va être utilisé pour créé des batchs afin d'entrainer le modèle
class NLUIntentDataset(Dataset):
    """
    Un Dataset Pytorch est un itérateur de tuple (couple) de tenseurs 
    composé de la source (utt encodée en token_id) et d'une cible.
    Pour la tâche de classification d'intent la cible est l'intent id
    Pour la tache de classification de slot la cible est une sequence de slot_id
    aussi longue que l'utt. 
    Cette classe construit son vocabulaire en séparant les mots par un ' '.
    Ce n'est pas appropiré pour les languages chinois et japonais.
    Args:
        Dataset (_type_): _description_

    Returns:
        _type_: _description_
    """
    padding = 1
     
    @staticmethod
    def add_tokens(utt:str,token2ind:dict):
        for word in utt.split(' '):
            if word not in token2ind.keys():
                if len(word)>0 and word not in token2ind.keys():
                    token2ind[word]=len(token2ind)
        return len(utt)
    
    @staticmethod
    def add_intent(intent:str,intent2ind:dict):
        if intent not in intent2ind.keys():
            intent2ind[intent]=len(intent2ind)

    
    def get_vocabulary_size(self)-> int :
        return len(self.token2ind)

    def __init__(self,json_files_list:list[str],btrain:bool):
        """
        
        Args:
            json_files_list (list[str]): liste de fichiers jsonl pris en compte pour le dataset
            btrain (bool): True, prend les utts de la partition train. False Prend les utt de la parition
            test.
        """
        super().__init__()
        self.intent_data = []
        self.token2ind = {"<sos>": 0, "<pad>": self.padding, "<eos>": 2, "<oov>": 3} #start of sentance : 0, padding : 1, end of sentance : 2, out of vocabulary : 3
        self.intent2ind = dict()
        self.utt_max_len = 0
        
        partition_filter = 'train' if btrain else 'test'
        #On extrait les données en ajoutant le partition filter
        for lang_jsfile in json_files_list:
            with open(lang_jsfile,'r') as lg_file:
                data = [json.loads(line) for line in lg_file if len(line.strip())>0 ]
                extracted = [ extract_intent_fields(jsobj) for jsobj in data if jsobj['partition']==partition_filter]
                self.intent_data.extend(extracted)
        #On créé un dictionnaire entre les tokens et leur id ; de même pour les intent
        for utt,intent in self.intent_data:
            utt_len = self.add_tokens(utt,self.token2ind)
            if utt_len > self.utt_max_len:
                self.utt_max_len = utt_len
            self.add_intent(intent,self.intent2ind)


    
    def __len__(self):
        return len(self.intent_data)

    def __getitem__(self, idx) -> dict[str,torch.Tensor]:
        """
        Args:
            idx (_type_): index du sample dans le dataset.

        Returns:
            dict[str,torch.Tensor]: Tenseur de long, les token_id et intent_id
        """
        utt,intent = self.intent_data[idx]
        utt_sequence = [self.token2ind["<sos>"]] + [
            self.token2ind[word] if word in self.token2ind.keys() else self.token2ind["<oov>"]
            for word in utt.split(' ')] + [self.token2ind["<eos>"]] #Une phrase commence par sos puis nos mots tokenisés ou non, puis la fin de phrase.
        intent_ind = [self.intent2ind[intent]]

        sample = {
            "source_sequence": torch.tensor(utt_sequence),
            "target": torch.tensor(intent_ind),
        }
        return sample
    

def get_loader(
  josnl_docs:list[str],
  btrain:bool,       
  batch_size:int=32
):
    dataset = NLUIntentDataset( #Il s'agit de l'itérateur précédemment défini
        josnl_docs,
        btrain=btrain,
 
    )
    data_loader = DataLoader( #Le data_loader est une classe pytorch qui sert à construire des batchs
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=MyCollator,
        pin_memory=True,
        drop_last=True,
    )
    return data_loader


def MyCollator(batch):
    source_sequences = pad_sequence(
        #on utilise le padding pour matcher la longueur des séquences dans le même batch
        [sample["source_sequence"] for sample in batch], padding_value= NLUIntentDataset.padding )
    target = [sample["target"] for sample in batch]
    return source_sequences, target # data[0] : doc, data[1]: label




class NLUIntentDatasetFastText(Dataset):
    """
    Dataset variante de NLUIntentDataset utilisant le vocabulaire
    contenu dans un modèle fasttext.
    """
    
    
    @staticmethod
    def add_intent(intent:str,intent2ind:dict):
        if intent not in intent2ind.keys():
            intent2ind[intent]=len(intent2ind)

    
    def get_vocabulary(self)-> dict[str,int] :
        """
        Word + Subwords as found in corpus

        Returns:
            dict[str,int]: _description_
        """
        return self.vocab
    
    def get_intent2id(self):
        return self.intent2ind

    def __init__(self,json_files_list:list[str],
                 btrain:bool,fasttext_model_path:str,vocab:dict[str,int],seq_len:int=-1):
        super().__init__()
        self.intent_data = []
        self.vocab = vocab
        self.fasttextModel = fasttext.load_model(fasttext_model_path)
        self.vocab_size = len(vocab)
        # L'entrée du token_id = 0 correspond à </s> dans un modèle
        # fasttext je l'ai utilisé pour le padding on peut faire mieux.
        self.padding_idx = 0

        self.intent2ind = dict()
        self.utt_max_len = 0
        
        partition_filter = 'train' if btrain else 'test'
        #On extrait les données en ajoutant le partition filter
        for lang_jsfile in json_files_list:
            with open(lang_jsfile,'r') as lg_file:
                data = [json.loads(line) for line in lg_file if len(line.strip())>0 ]
                extracted = [ extract_intent_fields(jsobj) for jsobj in data if jsobj['partition']==partition_filter ]
                self.intent_data.extend(extracted)
        #On créé un dictionnaire entre les tokens et leur id ; de même pour les intent
        for utt,intent in self.intent_data:
            self.add_intent(intent,self.intent2ind)
            utt_len = len(utt.strip())
            if utt_len > self.utt_max_len:
                self.utt_max_len = utt_len
        #pas de destruction d'information
        if seq_len >0 and seq_len < self.utt_max_len:
            raise ValueError(f"seq_len must be > {self.utt_max_len}")
        self.seq_eln = seq_len


    
    def __len__(self):
        return len(self.intent_data)

    def __getitem__(self, idx):
        """
        Utilise les token_id du modèle fasttext.
        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        utt,intent = self.intent_data[idx]
        utt_sequence = []
        for word in utt.strip().split(' '):
            subword_id = self.vocab[word]
            utt_sequence.append(subword_id)
            #subwords_f = [sub for sub in subwords if len(sub.strip())>0 and sub.isalpha()]
            #utt_sequence.extend([self.fasttextModel.get_subword_id(subw) for subw in subwords_f ])

        intent_ind = [self.intent2ind[intent]]

        sample = {
            "source_sequence": torch.tensor(utt_sequence),
            "target": torch.tensor(intent_ind),
        }
        return sample
    


def get_loader_fasttext(
  josnl_docs:list[str],
  btrain:bool, 
  fasttext_model_path:str,      
  vocab:dict[str,int],
  batch_size:int=32,
  seq_len:int=-1
):
    dataset = NLUIntentDatasetFastText( #Il s'agit de l'itérateur précédemment défini
        josnl_docs,
        btrain=btrain,
        fasttext_model_path=fasttext_model_path,
        vocab=vocab,
        seq_len=seq_len)
    
    #Un collator à préparer et regrouper nos données en un seul batch afin qu'ils soit formatés pour être traités par un modèle
    def MyCollatorFasttext(batch):
        #on utilise le padding pour matcher la longueur des séquences dans le même batch 
        if seq_len > 0:
            utt_0 = batch[0]["source_sequence"]
            #ce padding à utt_max_len assure des batch ayant tous la meme dimension [ N, utt_max_len]
            padded_batch0 = torch.nn.ConstantPad1d((0, dataset.seq_eln - utt_0.size(0)), dataset.padding_idx)(utt_0)
            batch[0]["source_sequence"] = padded_batch0
        source_sequences = pad_sequence(

            [sample["source_sequence"] for sample in batch], batch_first=True,
              padding_value= dataset.padding_idx )
        target =torch.stack([sample["target"] for sample in batch],dim=0).squeeze()
        return source_sequences, target # data[0] : doc, data[1]: label
    
    data_loader = DataLoader( #Le data_loader est une classe pytorch qui sert à construire des batchs
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=MyCollatorFasttext,
        pin_memory=True,
        drop_last=True,
    )
    return data_loader,dataset.seq_eln
    







#le "if __name__=main" sert à ce que notre compilateur l'execute directement. Fait office de fonction test.
if __name__ == "__main__":
    ds_path = join(os.path.dirname(os.path.realpath(__file__)),"massive_extract")
    models_dir = join(os.path.dirname(os.path.realpath(__file__)),"models")

    corpus_path =  join(os.path.dirname(os.path.realpath(__file__)),"corpuses","extract0_corpus.txt")

    fasttext_model_path = join(models_dir, "fst250_10K_all.model")
    model = fasttext.load_model(fasttext_model_path)
    vocab = get_vocab(corpus_path)

    files_list = ["de-DE.jsonl","en-US.jsonl","es-ES.jsonl","fr-FR.jsonl","it-IT.jsonl"]

    latin_dataset_files = [join(ds_path, f) for f in files_list]

    

    fasttextDS = NLUIntentDatasetFastText(latin_dataset_files,True,fasttext_model_path,vocab)
    intent2id = fasttextDS.get_intent2id()
    print(f"nb intents {len(intent2id)}")
    second_tensor = fasttextDS[1]
    
    
    vocab_fasttext = fasttextDS.get_vocabulary()
    nb_samples = len(fasttextDS)
    print(f"Fasttext Vocabulary size {len(vocab_fasttext)} nb samples {nb_samples}")
    

    my_loader_fast, seq_max_len = get_loader_fasttext(latin_dataset_files,True,fasttext_model_path,vocab,32)
    my_iter_fast = iter(my_loader_fast)
    batch_0,target_0 = next(my_iter_fast)
    print(batch_0.size())
    print(target_0.size())
    batch_1,target_1 = next(my_iter_fast)
    print(batch_1.size())


    

    my_intentDS = NLUIntentDataset(latin_dataset_files,btrain=True)
    vocab_size = my_intentDS.get_vocabulary_size()
    nb_samples = len(my_intentDS)
    print(f"Vocabulary size {vocab_size} nb samples {nb_samples}") # Lors de la première éxecution, on a une taille de vocabulaire 65900 et un nombre de phrases de 103620. 
    #On remarque que la taille du vocabulaire est trop grande par rapport au nombre de de phrases. Dès lors il nous parrait essentiel d'utiliser fastText pour diminuer la taille du vocabulaire.
    #second_tensor = my_intentDS[1]
    #print(second_tensor)

    #my_loader = get_loader([lang_jsfile],32)
    #my_iter = iter(my_loader)
    #batch_0,target_0 = next(my_iter)
    #print(batch_0.size())
    #batch_1,target_1 = next(my_iter)
    #print(batch_1.size())