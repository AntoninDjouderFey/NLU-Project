
from os import listdir
import os
from os.path import isfile, join
import json
import logging
import numpy as np
import fasttext
import unicodedata
import re
from collections import Counter
from matplotlib import pyplot as plt
import json


#Comme son nom l'indique cette fonction extrait du json un tuple composé de l'intent et de la phrase
def extract_intent_fields(json_obj:dict)-> str:
    if 'utt' not in json_obj.keys():
        raise ValueError("Invalid sample no UTT")
    norm_utt = normalize(json_obj['utt'])
    return norm_utt

def extract_all(json_obj:dict)-> tuple[str,str,str]:
    if 'annot_utt' not in json_obj.keys():
        raise ValueError("Invalid sample no annot_utt")
    norm_anot_utt = normalize(json_obj['annot_utt'])
    if 'utt' not in json_obj.keys():
        raise ValueError("Invalid sample no UTT")
    norm_utt = normalize(json_obj['utt'])
    if 'intent' not in json_obj.keys():
        raise ValueError("Invalid sample no intent")
    norm_intent = normalize(json_obj['intent'])
    return norm_utt,norm_intent,norm_anot_utt

def normalize(line):
    """Normalize a line of utf-8 text """
    line = unicodedata.normalize("NFKC", line.strip())
    return line

def process_annot_utt(annot_utt):
    """
    Finite state automata to process annot_utt:
    "bitte stell einen wecker für [date : morgen] um [time : sieben uhr dreißig morgens]"
    -->
    None None None None None date None time time time time
    """
    result = ""
    in_word = False
    in_slot = False
    current_type = ""
    in_current_type = False
    for l in annot_utt:
        if l.isspace():
            if in_word:
               in_word = False
               if not in_slot :
                   result = result + "None"
               else:
                   if in_current_type:
                       in_current_type=False
                   else:
                      result+=current_type
            if not in_current_type:
                if len(result)>0 and result[-1] != ' ':
                    result= result + ' '
        elif l.isalnum():
            if in_current_type:
                current_type= current_type + l
                continue
            if in_word:
                continue
            else:
                in_word = True
        elif l=='[':
            in_slot = True
            in_current_type = True
        elif l==']':
            in_slot = False
            in_current_type=False
            if in_word:
                result = result + current_type
                in_word = False
            current_type=''
        elif l==':':
            in_current_type = False
        elif l=='_':
            if in_current_type:
                current_type= current_type + l
                continue
        elif l in ['&','-','.','!','$','@','%']:
            in_word = True
            continue
    if in_word:
        result = result + "None"
    return result



#On extrait nos données dans un corpus
def extract_dataset(corpus_name,files_list:list[str],btrain=True):
    """
    Construit un ficher corpus contenant tous les utt avec un utt par ligne.
    Ce fichier sera utilisé en entrée de l'entrainement du modèle fasttext
    Args:
        corpus_name (_type_): nom du fichier corpus à créér
        files_list (list[str]): list des fichier jsonl à lire (contenu du répertoire massive_extract)
    Returns:
        int: longueur maximale des utts (utilisé pour construire des batch)
    """
    max_utt_len = 0
    

    with open(corpus_name,'w') as notre_corpus:

        #On extrait les données 
        dir_path = join(os.path.dirname(os.path.realpath(__file__)),"massive_extract") #créé le répertoire
        all_dataset_files = [join(dir_path, f) for f in files_list] #extrait les données de massives extract

        for lang_jsfile in all_dataset_files:
            with open(lang_jsfile,'r',encoding="utf-8") as lg_file:
                data = [json.loads(line) for line in lg_file if len(line.strip())>0 ]
                extracted = [ extract_intent_fields(jsobj) for jsobj in data  ]
                for utt in extracted:
                    notre_corpus.write(utt+"\n")
                    max_utt_len = len(utt) if len(utt)>max_utt_len else max_utt_len

    return max_utt_len


def create_preproc_dataset(dataset_name,files_list:list[str],btrain=True):
    """
    Construit un ficher Dataset preproc, chaque ligne de ce fichier text :
    utt|intent|utt_slots, par exemple
    play magic run after thirty min|play_audiobook|None audiobook_name audiobook_name time time time
    Args:
        dataset_name (_type_): nom du fichierdataset_name à créér
        files_list (list[str]): list des fichier jsonl à lire (contenu du répertoire massive_extract
        btrain : True selectionne la partition train, False partition de test
    Returns:
        int: longueur maximale des utts (utilisé pour construire des batch)
    """
    max_utt_len = 0
    partition_filter = 'train' if btrain else 'test'
    with open(dataset_name,'w') as notre_corpus:

        #On extrait les données 
        dir_path = join(os.path.dirname(os.path.realpath(__file__)),"massive_extract") #créé le répertoire
        all_dataset_files = [join(dir_path, f) for f in files_list] #extrait les données de massives extract

        for lang_jsfile in all_dataset_files:
            with open(lang_jsfile,'r',encoding="utf-8") as lg_file:
                data = [json.loads(line) for line in lg_file if len(line.strip())>0 ]
                utt_intent_anot = [ extract_all(jsobj) for jsobj in data if jsobj['partition']==partition_filter ]
                for utt,intent,utt_anot in utt_intent_anot:
                    slotted_anot = process_annot_utt(utt_anot)
                    notre_corpus.write(f"{utt}|{intent}|{slotted_anot}\n")
                    max_utt_len = len(utt) if len(utt)>max_utt_len else max_utt_len

    return max_utt_len

def add_intent(intent2ind:dict[str,int],intent:str):
        if intent not in intent2ind.keys():
            intent2ind[intent]=len(intent2ind)
   
def add_slots(slot2ind,utt_slots:str):       
    for slot in utt_slots.split(' '):
        slot = slot.strip()
        if slot not in slot2ind.keys() and len(slot)>0:
            slot2ind[slot] = len(slot2ind)
            

def create_intent2ind_slot2int(proc_ds_path_list:list[str]):
    """
    Construit les mapping intent -> id et slot -> id
    avec slot = date, time, artist_name, None
    A partir de dataset preproc créés par create_preproc_dataset
    """
    intent2id = dict()
    slot2id = dict()
    slot2id['None'] = 0
    for path in proc_ds_path_list:
        with open(path,'r',encoding="utf-8") as ds_file:
            for line in ds_file:
                if len(line.strip())>0:
                    utt,intent,slots = line.strip().split('|')
                    add_intent(intent2id,intent)
                    add_slots(slot2id,slots)

    return intent2id,slot2id


def compute_class_weights(intent2id:dict[str,int],slot2id:dict[str,int],proc_ds_path_list):
    """
    Computes intent and slot classes weights for balanced classification, for each intent_id or slot_id sorted by ascending order 
    [weight_for_classid_0,weight_for_class_id_1,...,weight_for_class_id_N]
    each weight is total_nb_samples / (nb_classes * nb_samples_for_class_i ) ; to make sure that no class is overrepresented
    Similar to sklearn  n_samples / (n_classes * np.bincount(y))
    """
    intent_counter = Counter() #Un counter est un dictionnaire pour lesquels les valeurs sont seulement des entiers
    slot_counter = Counter()
    for path in proc_ds_path_list:
        with open(path,'r',encoding="utf-8") as ds_file:
            for line in ds_file:
                if len(line.strip())>0:
                    utt,intent,slots = line.strip().split('|')
                    intent_id = intent2id[intent]
                    intent_counter[intent_id]+=1
                    for slot in slots.split(' '):
                        if len(slot.strip())>0:
                            slot_id = slot2id[slot]
                            slot_counter[slot_id]+=1
    total_intents = sum(intent_counter.values())
    nb_intents = len(intent2id)
    nb_slots = len(slot2id)
    total_slots = sum(slot_counter.values())
    intent_weight_list = [ round(total_intents/ (nb_intents*intent_counter[id]),3) for id in sorted(intent_counter.keys()) ]
    slot_weight_list = [ round(total_slots/(nb_slots*slot_counter[id]),3) for id in sorted(slot_counter.keys()) ]
    return intent_weight_list,slot_weight_list

def get_counters(intent2id:dict[str,int],slot2id:dict[str,int],proc_ds_path_list): #Le dernier argument contient le chemin des datasets préprocessés
    intent_counter = Counter() #Un counter est un dictionnaire pour lesquels les valeurs sont seulement des entiers
    slot_counter = Counter()
    for path in proc_ds_path_list:
        with open(path,'r',encoding="utf-8") as ds_file:
            for line in ds_file:
                if len(line.strip())>0:
                    utt,intent,slots = line.strip().split('|')
                    intent_id = intent2id[intent]
                    intent_counter[intent_id]+=1
                    for slot in slots.split(' '):
                        if len(slot.strip())>0:
                            slot_id = slot2id[slot]
                            slot_counter[slot_id]+=1
    return intent_counter,slot_counter



def fasttext_model_construction(corpus,vector_size:int,model_name:str):
    """
    
    Args:
        corpus (_type_): Le fichier contenant tous les utt avec un utt par ligne
        vector_size (int): dimension des vecteurs (Embedding) representant les mots ou sous-mots
        model_name (str): path du modele fasttext sauvegradé

    Returns:
        _type_: _description_
    """
    model= model = fasttext.train_unsupervised(corpus, model='skipgram',dim=vector_size,neg=3,epoch=20,wordNgrams=2)
    # build the vocabulary
    model_path =  join(os.path.dirname(os.path.realpath(__file__)),"models",model_name)
    model.save_model(model_path)
    print(f"Model vocab size {len(model.words)}")
    return model_path

def get_vocab(corpus_path) -> dict[str,int]:
    """
    Renvoie un mapping word/subword --> id ou token_id
    Returns:
        _type_: _description_
    """
    result = dict()
    with open(corpus_path,'r') as crp :
       for line in crp:
           if len(line.strip())>0:
               words = line.strip().split(' ')
               for wd in words:
                   if wd not in result.keys():
                       result[wd] = len(result)
    return result
                   

def plotter(any_counter:Counter):
    x=np.array(list(any_counter.keys()),dtype=np.float64)

    y=np.array(list(any_counter.values()),dtype=np.float64)
    plt.bar(x,y)
    plt.xlabel("Nos intents/slots")
    plt.ylabel("Leur valeurs")
    plt.title("Fréquence de nos intents/slots")
    plt.show()    
    
if __name__=='__main__':

    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    corpus_path =  join(os.path.dirname(os.path.realpath(__file__)),"corpuses","extract0_corpus.txt")
    train_nofr_dataset_path =  join(os.path.dirname(os.path.realpath(__file__)),"datasets","train_nofr_formatted_dataset.txt")
    test_nofr_dataset_path =  join(os.path.dirname(os.path.realpath(__file__)),"datasets","test_nofr_formatted_dataset.txt")
    fr_eval_dataset_path = join(os.path.dirname(os.path.realpath(__file__)),"datasets","fr_eval_dataset.txt")

    #intent2id,slot2id= create_intent2ind_slot2int([train_dataset_path,test_dataset_path])

    #intent_counter,slot_counter=get_counters(intent2id,slot2id,[train_dataset_path,test_dataset_path])
    #plotter(intent_counter)
    #plt.savefig("intent_histogram")
    #plotter(slot_counter)
    #plt.savefig("slot_histogram")



    # on travaille d'abord avec les fichiers de langue lantine qui peuvent avoir des
    # subword fasttext en commun
    #files_list = ["de-DE.jsonl","en-US.jsonl","es-ES.jsonl","fr-FR.jsonl","it-IT.jsonl"]
    files_list_no_fr = ["de-DE.jsonl","en-US.jsonl","es-ES.jsonl","it-IT.jsonl"]
    file_list_fr = ["fr-FR.jsonl"]
    
    #max_utt_len = extract_dataset(corpus_path,files_list=files_list)
    max_utt_len = create_preproc_dataset(train_nofr_dataset_path,files_list=files_list_no_fr)
    max_utt_len = create_preproc_dataset(test_nofr_dataset_path,files_list=files_list_no_fr,btrain=False)
    max_utt_len = create_preproc_dataset(fr_eval_dataset_path,files_list=file_list_fr)
    #print(f"Max utt len is {max_utt_len}")

    #max_utt_len = create_preproc_dataset(test_dataset_path,files_list=files_list,btrain=False)
    #print(f"Max utt len is {max_utt_len}")
    # training / contruction du modèle fasstext
    model_path = fasttext_model_construction(corpus_path,100,"fst100_all.model")
    #model_path = "/home/rdr/antonin_nlu/models/fst250_10K_all.model"
    #model = fasttext.load_model(model_path)
    #vocab_sz = len(model.words)
    # l'input matrix contient tous les vecteur representant les mots & sous-mots
    #mat = model.get_input_matrix()
    #print(mat.shape)

    #le vocabulaire réel construit par fasttext avec les sous-mots



    # Load back the same model. 
    #loaded_model = FastText.load(tmp.name)
    #print(loaded_model)
    