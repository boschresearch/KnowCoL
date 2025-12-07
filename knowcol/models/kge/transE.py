import torch
import torch.nn as nn
from knowcol.datasets.kg import KG
import pickle
import numpy as np
from pathlib import Path
import knowcol.models.knowcol as knowcol
from tqdm import tqdm
import torch.nn.functional as F
import logging
logger = logging.getLogger(__name__)

class TransE(nn.Module):
    def __init__(self, latent_dim:int, kg: KG, pretrained_model_path: str = None):
        super(TransE, self).__init__()
        node_embs = torch.Tensor(np.random.uniform(low=-6.0/np.sqrt(latent_dim), high=6.0/np.sqrt(latent_dim), size=(kg.n_ent+1, latent_dim)))
        rel_embs = torch.Tensor(np.random.uniform(low=-6.0/np.sqrt(latent_dim), high=6.0/np.sqrt(latent_dim), size=(kg.n_rel+1, latent_dim)))
        
        self.n_ent = kg.n_ent
        self.n_rel = kg.n_rel
        if pretrained_model_path:
            # load pretrained embbedings from pretrained knowledge graph embeddings
            if not Path(pretrained_model_path).is_absolute():
                pretrained_model_path = Path(knowcol.__file__).parent.parent / pretrained_model_path
            with open(pretrained_model_path, "rb") as inflie:
                pretrained_model = pickle.load(inflie)
            ent2ix_p = pretrained_model.graph.entity2id
            rel2ix_p = pretrained_model.graph.relation2id
            node_embs_p = pretrained_model.solver.entity_embeddings
            rel_embs_p = pretrained_model.solver.relation_embeddings

            logger.info('loading pretrained entity embeddings')
            for k in tqdm(kg.ent2ix):
                if k in ent2ix_p:
                    node_embs[kg.ent2ix[k]] = torch.tensor(node_embs_p[ent2ix_p[k]])

            logger.info('loading pretrained relation embeddings')
            for k in tqdm(kg.rel2ix):
                if k in rel2ix_p:
                    rel_embs[kg.rel2ix[k]] = torch.tensor(rel_embs_p[rel2ix_p[k]])

        self.node_embs = nn.Embedding.from_pretrained(node_embs, padding_idx=0, freeze=False)
        self.rel_embs = nn.Embedding.from_pretrained(rel_embs, padding_idx=0, freeze=False)
    
    def get_node_embs(self, entity_ix: torch.Tensor):
        return self.node_embs(entity_ix)
    
    def get_all_node_embs(self):
        return self.node_embs.weight[1:,:] # (N_ent, N_z) ignore the first embeddings for padding


    def forward(self, triplets):
        '''
            Args:
                triplets: with the size of (*, 3)
            return:
                head entity embedding (*, N_z), relation embedding (*, N_z), tail entity embedding (*, N_z)
        '''
        return self.node_embs(triplets[..., 0]), self.rel_embs(triplets[..., 1]), self.node_embs(triplets[...,2])

        
        