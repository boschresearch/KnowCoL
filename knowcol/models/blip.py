import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl 
from omegaconf import DictConfig
import pdb
import open_clip
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import json

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data

class BLIP(pl.LightningModule):
    def __init__(self, pre_trained_model, wikipedia_entity_info_file):
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
        )  # doctest: +IGNORE_RESULT


        wikipedia_entity_info = load_json(wikipedia_entity_info_file)

        self.name2entityid = {}
        for d in wikipedia_entity_info:
            self.name2entityid[d["id"]] = d["wikipedia_title"]
        self.corpus = [
            d["wikipedia_title"] for d in self.wikipedia_entity_info
        ]
        tokenized_corpus = [self.tokenizer.tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
    def inference(self, inputs):
        answers = self.model.generate({"image": batch, "prompt": prompt},
                                 length_penalty=-1) # (B, )
        tokenized_answers = self.tokenizer.tokenize(answers)
        names = self.bm25.get_top_n(tokenized_answers, self.corpus, n=1) # (B,)
        entity_ids = [self.name2entityid[n] for n in names]
        return entity_ids
    

 