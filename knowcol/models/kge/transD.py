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
import pdb
logger = logging.getLogger(__name__)

class TransD(nn.Module):
    def __init__(self, latent_dim: int, kg, pretrained_model_path: str = None):
        super(TransD, self).__init__()
        self.n_ent = kg.n_ent
        self.n_rel = kg.n_rel
        self.emb_dim = latent_dim
        self.ce_loss_none = nn.CrossEntropyLoss(reduction='none')

        # Entity + relation embeddings & mapping vectors
        ent_init = torch.Tensor(np.random.uniform(
            low=-6.0 / np.sqrt(latent_dim),
            high=6.0 / np.sqrt(latent_dim),
            size=(self.n_ent + 1, latent_dim)
        ))
        rel_init = torch.Tensor(np.random.uniform(
            low=-6.0 / np.sqrt(latent_dim),
            high=6.0 / np.sqrt(latent_dim),
            size=(self.n_rel + 1, latent_dim)
        ))
        ent_map_init = torch.Tensor(np.random.uniform(
            low=-6.0 / np.sqrt(latent_dim),
            high=6.0 / np.sqrt(latent_dim),
            size=(self.n_ent + 1, latent_dim)
        ))
        rel_map_init = torch.Tensor(np.random.uniform(
            low=-6.0 / np.sqrt(latent_dim),
            high=6.0 / np.sqrt(latent_dim),
            size=(self.n_rel + 1, latent_dim)
        ))

        # Optional: load pretrained weights
        # if pretrained_model_path:
        #     if not Path(pretrained_model_path).is_absolute():
        #         pretrained_model_path = Path(knowcol.__file__).parent.parent / pretrained_model_path
        #     with open(pretrained_model_path, "rb") as infile:
        #         pretrained_model = pickle.load(infile)

        #     ent2ix_p = pretrained_model.graph.entity2id
        #     rel2ix_p = pretrained_model.graph.relation2id
        #     ent_embs_p = pretrained_model.solver.entity_embeddings
        #     rel_embs_p = pretrained_model.solver.relation_embeddings
        #     ent_map_p = getattr(pretrained_model.solver, "entity_mapping", None)
        #     rel_map_p = getattr(pretrained_model.solver, "relation_mapping", None)

        #     logger.info("Loading pretrained entity embeddings...")
        #     for k in tqdm(kg.ent2ix):
        #         if k in ent2ix_p:
        #             ent_init[kg.ent2ix[k]] = torch.tensor(ent_embs_p[ent2ix_p[k]])

        #     logger.info("Loading pretrained relation embeddings...")
        #     for k in tqdm(kg.rel2ix):
        #         if k in rel2ix_p:
        #             rel_init[kg.rel2ix[k]] = torch.tensor(rel_embs_p[rel2ix_p[k]])
        #             if rel_map_p is not None:
        #                 rel_map_init[kg.rel2ix[k]] = torch.tensor(rel_map_p[rel2ix_p[k]])
        #     if ent_map_p is not None:
        #         for k in tqdm(kg.ent2ix):
        #             if k in ent2ix_p:
        #                 ent_map_init[kg.ent2ix[k]] = torch.tensor(ent_map_p[ent2ix_p[k]])

        # Embeddings
        self.ent_embs = nn.Embedding.from_pretrained(ent_init, padding_idx=0, freeze=False)
        self.ent_maps = nn.Embedding.from_pretrained(ent_map_init, padding_idx=0, freeze=False)
        self.rel_embs = nn.Embedding.from_pretrained(rel_init, padding_idx=0, freeze=False)
        self.rel_maps = nn.Embedding.from_pretrained(rel_map_init, padding_idx=0, freeze=False)

    def project(self, e_emb: torch.Tensor, e_map: torch.Tensor, r_map: torch.Tensor):
        """
        TransD dynamic projection
        e_emb: (*, dim)
        e_map: (*, dim)
        r_map: (*, dim)
        return: (*, dim)
        """
        original_shape = e_emb.shape  # (..., dim)
        dim = original_shape[-1]
        flat_e_emb = e_emb.view(-1, dim)     # (N, dim)
        flat_e_map = e_map.view(-1, dim)     # (N, dim)
        flat_r_map = r_map.view(-1, dim)     # (N, dim)

        # Outer product and add identity
        outer = flat_r_map.unsqueeze(-1) * flat_e_map.unsqueeze(-2)  # (N, dim, dim)
        eye = torch.eye(dim, device=e_emb.device).unsqueeze(0)       # (1, dim, dim)
        proj_matrix = outer + eye                                    # (N, dim, dim)

        # Project entity embedding
        flat_proj = torch.bmm(proj_matrix, flat_e_emb.unsqueeze(-1)).squeeze(-1)  # (N, dim)

        # Reshape back
        return flat_proj.view(*original_shape)  # (..., dim)

    def forward(self, triplets: torch.Tensor):
        """
        Args:
            triplets: (*, 3)
        Returns:
            projected_h, r, projected_t
        """
        h_idx, r_idx, t_idx = triplets[..., 0], triplets[..., 1], triplets[..., 2]

        h = self.ent_embs(h_idx)
        t = self.ent_embs(t_idx)
        r = self.rel_embs(r_idx)

        p_h = self.ent_maps(h_idx)
        p_t = self.ent_maps(t_idx)
        p_r = self.rel_maps(r_idx)

        h_proj = self.project(h, p_h, p_r)
        t_proj = self.project(t, p_t, p_r)

        return h_proj, r, t_proj    

    def get_node_embs(self, entity_ix: torch.Tensor):
        return self.ent_embs(entity_ix)

    def get_all_entity_embeddings(self):
        return self.ent_embs.weight[1:, :]