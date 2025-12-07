import torch.nn as nn
import pytorch_lightning as pl 
from omegaconf import DictConfig
import hydra
import numpy as np
import logging
import torch
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple
from .triplet import Triplet
import knowcol
import pdb
logger = logging.getLogger(__name__)

def to_absolute_path(path: str):
    path = Path(path)
    if not path.is_absolute():
        path = Path(knowcol.__file__).parent.parent / path
    return path
    
class KG():
    def __init__(self, knowledge_base_path: str, entity_set_path: str, relation_path: str, triplet_h_path: str, triplet_t_path: str):
        super(KG, self).__init__()
        self.kb_path = to_absolute_path(knowledge_base_path)

        self.entity_set_path = to_absolute_path(entity_set_path)
        self.relation_path = to_absolute_path(relation_path)
        self.triplet_h_path = to_absolute_path(triplet_h_path)
        self.triplet_t_path = to_absolute_path(triplet_t_path)

        self.ent2ix = {}
        self.rel2ix = {}
        self.entities = []
        self.relations = []
        self.triplets_h_dict = {}
        self.triplets_t_dict = {}
        self.ent_info = {} # {'id': {'img': 'path_to_image', 'text': 'xxx'}}
        self.rel_info = {} # {'id': 'text description xxx'}
        self._load_entity()
        self._load_relation()
        self._load_triples()
        self.n_ent = len(self.ent2ix)
        self.n_rel = len(self.rel2ix)
        
        
        # self.node_embs = nn.Parameter(torch.tensor(np.random.uniform(low=-6.0/np.sqrt(latent_dim), high=6.0/np.sqrt(latent_dim), size=(self.n_ent, latent_dim))))
        # self.rel_embs = nn.Parameter(torch.tensor(np.random.uniform(low=-6.0/np.sqrt(latent_dim), high=6.0/np.sqrt(latent_dim), size=(self.n_rel, latent_dim))))

    def _qid2img(self, qid: str):
        id = int(qid[1:])
        if id < 100:
            return (self.kb_path / 'wikipedia_images_full' / qid / f'{qid}.jpg').as_posix()
        else:
            return (self.kb_path / 'wikipedia_images_full' / qid[:4] / f'{qid}.jpg').as_posix()


    def _load_entity(self):
        ### qid to index 
        with open(self.entity_set_path, 'r') as infile:
            logger.info('Loading entities...')
            for i, f in enumerate(tqdm(infile)):
                self.ent2ix[f[:-1]] = i+1 # remove \n at the end of line, ignore the 0th index, because the 0th index is used for padding
                self.entities.append(f[:-1])
            logger.info('Loading entities finished!')

        ### qid to img path and text description
        with open(self.kb_path / 'Wiki6M_ver_1_1.jsonl', 'r') as infile:
            logger.info('Loading entities context...')
            for f in tqdm(infile):
                json_obj = json.loads(f)
                if json_obj['wikidata_id'] in self.ent2ix:
                    self.ent_info[json_obj['wikidata_id']]= {'img': self._qid2img(json_obj['wikidata_id']), 'text':json_obj['wikipedia_summary']}
            logger.info('Loading entities context finished!')

    def _load_relation(self):
        ### pid to index
        with open(self.relation_path, 'r') as infile:
            logger.info('Loading relations...')
            for i, f in enumerate(tqdm(infile)):
                self.rel2ix[f[:-1]] = i+1  # remove \n at the end of line, ignore the 0th index, because the 0th index is used for padding
                self.relations.append(f[:-1])
                
        ### pid to text description
        with open(self.kb_path / 'wikidata_relation_1_1.jsonl', 'r') as infile:
            logger.info('Loading relations description...')
            for f in tqdm(infile):
                json_obj = json.loads(f)
                self.rel_info[json_obj['p_id']] = json_obj['description']

            
    def _load_triples(self):
        with open(self.triplet_h_path, 'r') as infile:
            logger.info('Loading triplets...')
            for f in tqdm(infile):
                json_object = json.loads(f)
                self.triplets_h_dict[json_object['key']] = json_object['triplets']
            
        with open(self.triplet_t_path, 'r') as infile:
            for f in tqdm(infile):
                json_object = json.loads(f)
                self.triplets_t_dict[json_object['key']] = json_object['triplets']
            logger.info('Loading triplets finished!')
    
    def get_triplets_h(self, qid: str) -> List[Triplet]:
        '''
            Args: 
                qid: the entity id in the wikidata
            Return:
                triplets: a list containing [h,r,t]), indexed by pid and qid.
        '''
        if qid in self.triplets_h_dict:
            triplets_id = self.triplets_h_dict[qid]
        else:
            triplets_id = []
        triplets = list(map(lambda t: 
                               Triplet([self.ent2ix[t[0]], self.rel2ix[t[1]], self.ent2ix[t[2]]], self.rel_info[t[1]], direction='out'), 
                               triplets_id))
        return triplets

    def get_triplets_t(self, qid: str) -> List[Triplet]:
        '''
            Args: 
                qid: the entity id in the wikidata
            Return:
                triplets: a list containing [h,r,t]), indexed by pid and qid.
        '''
        if qid in self.triplets_t_dict:
            triplets_id = self.triplets_t_dict[qid]
        else:
            triplets_id = []
        triplets = list(map(lambda t: 
                               Triplet([self.ent2ix[t[0]], self.rel2ix[t[1]], self.ent2ix[t[2]]], self.rel_info[t[1]], direction='in'), 
                               triplets_id))
        return triplets

    def get_rels_text(self, pids: List[str]) -> List[str]:
        '''
            Args: 
                pids: a list of pids
            Return:
                relation description: a list of string
        '''
        return list(map(lambda s: self.rel_info[s], pids))
    
    def get_ents_text(self, qids: List[str]) -> List[str]:
        return list(map(lambda s: self.ent_info[s]['text'] if s in self.ent_info else 'None', qids))
    
    def get_ents_image(self, qids: List[str]) -> List[str]:
        return list(map(lambda s: self.ent_info[s]['img'] if s in self.ent_info else 'None', qids))
    
    def get_rel_text(self, pid: str) -> str:
        '''
            Args: 
                pid: a property (relation) id of wikidata
            Return:
                a description of property (relation)
        '''
        return self.rel_info[pid]

    def get_ent_text(self, qid: str) -> str:
        '''
            Args: 
                pid: a entity id of wikidata
            Return:
                a description of entity
        '''
        return self.ent_info[qid]['text']
    
    def get_ent_image(self, qid: str) -> str:
        '''
            Args: 
                pid: a entity id of wikidata
            Return:
                the path of an image of this entity
        '''
        return self.ent_info[qid]['img']