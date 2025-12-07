import torch
import torch.nn as nn
from knowcol.datasets.kg import KG
import pickle
import numpy as np
from pathlib import Path
import knowcol.models.knowcol as knowcol
from tqdm import tqdm
import logging
import torch.nn.functional as F
logger = logging.getLogger(__name__)


class TransH(nn.Module):
    def __init__(self, latent_dim: int, kg, pretrained_model_path: str = None):
        super(TransH, self).__init__()
        self.n_ent = kg.n_ent
        self.n_rel = kg.n_rel
        self.latent_dim = latent_dim
        self.ce_loss_none = nn.CrossEntropyLoss(reduction='none')

        # Initialize entity embeddings
        ent_init = torch.Tensor(np.random.uniform(
            low=-6.0 / np.sqrt(latent_dim),
            high=6.0 / np.sqrt(latent_dim),
            size=(self.n_ent + 1, latent_dim)
        ))
        # Initialize relation translation vectors
        rel_d_init = torch.Tensor(np.random.uniform(
            low=-6.0 / np.sqrt(latent_dim),
            high=6.0 / np.sqrt(latent_dim),
            size=(self.n_rel + 1, latent_dim)
        ))
        # Initialize relation hyperplane normal vectors
        rel_w_init = torch.Tensor(np.random.uniform(
            low=-6.0 / np.sqrt(latent_dim),
            high=6.0 / np.sqrt(latent_dim),
            size=(self.n_rel + 1, latent_dim)
        ))

        # Load pretrained embeddings if provided
        if pretrained_model_path:
            if not Path(pretrained_model_path).is_absolute():
                pretrained_model_path = Path(knowcol.__file__).parent.parent / pretrained_model_path
            with open(pretrained_model_path, "rb") as infile:
                pretrained_model = pickle.load(infile)

            ent2ix_p = pretrained_model.graph.entity2id
            rel2ix_p = pretrained_model.graph.relation2id
            node_embs_p = pretrained_model.solver.entity_embeddings
            rel_embs_p = pretrained_model.solver.relation_embeddings
            rel_normals_p = pretrained_model.solver.relation_normals if hasattr(pretrained_model.solver, "relation_normals") else rel_embs_p

            logger.info("Loading pretrained entity embeddings...")
            for k in tqdm(kg.ent2ix):
                if k in ent2ix_p:
                    ent_init[kg.ent2ix[k]] = torch.tensor(node_embs_p[ent2ix_p[k]])

            logger.info("Loading pretrained relation embeddings...")
            for k in tqdm(kg.rel2ix):
                if k in rel2ix_p:
                    rel_d_init[kg.rel2ix[k]] = torch.tensor(rel_embs_p[rel2ix_p[k]])
                    rel_w_init[kg.rel2ix[k]] = torch.tensor(rel_normals_p[rel2ix_p[k]])

        # Define trainable embeddings
        self.ent_embs = nn.Embedding.from_pretrained(ent_init, padding_idx=0, freeze=False)
        self.rel_d_embs = nn.Embedding.from_pretrained(rel_d_init, padding_idx=0, freeze=False)
        self.rel_w_embs = nn.Embedding.from_pretrained(rel_w_init, padding_idx=0, freeze=False)

    def project(self, entity_emb: torch.Tensor, norm_vector: torch.Tensor):
        """
        Projects entity embeddings onto the relation-specific hyperplane.
        Args:
            entity_emb: (*, dim)
            norm_vector: (*, dim)
        Returns:
            projected_entity: (*, dim)
        """
        norm_vector = nn.functional.normalize(norm_vector, p=2, dim=-1)
        return entity_emb - (torch.sum(entity_emb * norm_vector, dim=-1, keepdim=True) * norm_vector)

    def forward(self, triplets: torch.Tensor):
        """
        Args:
            triplets: Tensor of shape (*, 3) representing (head, relation, tail)
        Returns:
            h_proj: head entity projected (*, dim)
            r_trans: relation translation vectors (*, dim)
            t_proj: tail entity projected (*, dim)
        """
        h = self.ent_embs(triplets[..., 0])
        d = self.rel_d_embs(triplets[..., 1])
        t = self.ent_embs(triplets[..., 2])
        w = self.rel_w_embs(triplets[..., 1])

        # Normalize relation normal
        w_norm = F.normalize(w, p=2, dim=-1)
        # Gram-Schmidt: project translation vector d to be orthogonal to w_norm
        r_proj = d - torch.sum(d * w_norm, dim=-1, keepdim=True) * w_norm

        # Project head and tail onto the relation hyperplane
        h_proj = self.project(h, w_norm)
        t_proj = self.project(t, w_norm)

        return h_proj, r_proj, t_proj
    
    def get_node_embs(self, entity_ix: torch.Tensor):
        return self.ent_embs(entity_ix)
    
    def get_all_entity_embeddings(self):
        return self.ent_embs.weight[1:, :]

        
        