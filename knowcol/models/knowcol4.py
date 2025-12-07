import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl 
from omegaconf import DictConfig
import hydra
import torch.nn.functional as F
from knowcol.datasets.kg import KG
from knowcol.evaluations.evaluation import evaluate_oven_full
from knowcol.datasets.data_module import _transform
from knowcol.datasets.utils import _load_image_from_path
from knowcol.datasets.datasets import OvenKnowledgeBaseDataset
import pdb
import open_clip
from tqdm import tqdm
N_px = 224

    
class KnowCoL(pl.LightningModule):
    def __init__(self, encoder: DictConfig, kge: DictConfig, optimizer: DictConfig, beta1: int, beta2: int, temperature: float, kg: KG, lr_scheduler: DictConfig=None):
        super(KnowCoL, self).__init__()
        self.encoder_cfg = encoder
        self.clip_encoder = hydra.utils.instantiate(encoder)
        self.kge = hydra.utils.instantiate(kge, kg=kg)
        self.kg = kg
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.beta1 = beta1
        self.beta2 = beta2
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.ce_loss_none = nn.CrossEntropyLoss(reduction='none')
        self.test_outputs = [[],[]] # [entity, query]
        self.save_hyperparameters()

    def _cos_sim(self, e1: torch.Tensor, e2: torch.Tensor, t: float):
        '''
            input:
                e1: first input embedding with the form of (*, N_z)
                e2: second input embedding with the form of (*, N_z)
            output:
                cosine similiarity: with the form of (*)
        '''
        e1 = F.normalize(e1, dim=-1) # (*, N_z)
        e2 = F.normalize(e2, dim=-1) # (*, N_z)

        return torch.sum(e1 * e2, dim=-1) / t# (*)
    
    def _cos_sim_matrix(self, e1: torch.Tensor, e2: torch.Tensor, t: float):
        '''
            input:
                e1: first input embedding with the form of (B, N_z)
                e2: second input embedding with the form of (B, N_z)
                t: temperature
            output:
                cosine similiarity: with the form of (B, B)
        '''
        e1 = F.normalize(e1, dim=-1) # (B, N_z)
        e2 = F.normalize(e2, dim=-1) # (B, N_z)
        logits = e1 @ e2.t() / t
        return logits


    def _ke_loss(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor, masks: torch.Tensor):
        '''
            input:
                h: head embeddings with the shape of (B, N_nt_num+1, N_pt, N_z)
                r: relation embeddings with the shape of (B, N_nt_num+1, N_pt, N_z)
                t: tail embeddings with the shape of (B, N_nt_num+1, N_pt, N_z)
                masks: masks of the shape of (B, N_pt)
            output:
                cross_entropy_loss
        '''
        logits = self._cos_sim(h+r, t, self.temperature) # (B, N_nt_num+1, N_pt)
        ## TODO: decide which ke_loss to use
        labels = torch.zeros(logits.size(0), logits.size(2), device=self.device, dtype=torch.long) # (B, N_pt) the 0th triplet is the positive one
        loss = self.ce_loss_none(logits, labels) # (B, N_pt)
        return torch.sum(loss * masks) / torch.sum(masks)

    # def _ke_loss(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor, masks: torch.Tensor):
    #     '''
    #         input:
    #             h: head embeddings with the shape of (B, N_nt_num+1, N_pt, N_z)
    #             r: relation embeddings with the shape of (B, N_nt_num+1, N_pt, N_z)
    #             t: tail embeddings with the shape of (B, N_nt_num+1, N_pt, N_z)
    #             masks: masks of the shape of (B, N_nt_num+1, N_pt)
    #         output:
    #             cross_entropy_loss
    #     '''

    #     ## TODO: decide which ke_loss to use
    #     # labels = torch.zeros(logits.size(0), logits.size(2), device=self.device, dtype=torch.long) # (B, N_pt) the 0th triplet is the positive one
    #     # self.ce_loss(logits, labels)
    #     logits = self._cos_sim(h+r, t, self.temperature) # (B, N_nt_num+1, N_pt)
    #     pos_masks = (masks == 1).long()
    #     neg_masks = (masks == -1).long()
    #     # logits = logits * masks
    #     # return -1.0 * torch.mean(logits)
    #     return -1.0 * (torch.sum(pos_masks * logits)/torch.sum(pos_masks) - torch.sum(neg_masks * logits)/torch.sum(neg_masks))

    #     # logits = self._cos_sim(h+r, t, self.temperature) # (B, N_nt_num+1, N_pt)
    #     # labels = torch.zeros(logits.size(0), logits.size(2), device=self.device, dtype=torch.long) # (B, N_pt) the 0th triplet is the positive one
    #     # return self.ce_loss(logits, labels)

    # def _ke_loss(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor, masks: torch.Tensor):
    #     '''
    #         input:
    #             h: head embeddings with the shape of (B, N_nt_num+1, N_pt, N_z)
    #             r: relation embeddings with the shape of (B, N_nt_num+1, N_pt, N_z)
    #             t: tail embeddings with the shape of (B, N_nt_num+1, N_pt, N_z)
    #             masks: masks of the shape of (B, N_nt_num+1, N_pt)
    #         output:
    #             cross_entropy_loss
    #     '''

    #     logits = self._cos_sim(h+r, t, self.temperature) # (B, N_nt_num+1, N_pt)
    #     pos_masks = (masks == 1).long()  # (B, N_nt_num+1, N_pt)
    #     neg_masks = (masks == -1).long()  # (B, N_nt_num+1, N_pt)
    #     loss = 0.0
    #     for i in range(logits.shape[1]-1): # 0 to N_nt_num
    #         loss += torch.sum(torch.clamp(pos_masks[:,0,:] * logits[:,0,:] - neg_masks[:,i+1,:] * logits[:,i+1,:], max=1.0/self.temperature)) / torch.sum(pos_masks[:,0,:]) 
    #     return -1.0 * loss / (logits.shape[1]-1)


    def _prototype_alignment_loss(self, entity_embs, entity_image_embs, entity_text_embs):
        '''
            input:
                entity_embs: the node embeddings of entity with the shape of (B, N_z)
                entity_image_embs: the image embeddings of the entity with the shape of (B, N_z)
                entity_text_embs: the text embeddings of the entity with the shape of (B, N_z)
            output:
                prototype_alignment_loss
        '''
        logits = self._cos_sim_matrix(entity_embs, entity_image_embs, self.temperature) # (B, B)
        labels = torch.arange(logits.size(0), device=self.device)
        prototype_image_loss = (self.ce_loss(logits, labels) + self.ce_loss(logits.t(), labels)) / 2.0
        
        logits = self._cos_sim_matrix(entity_embs, entity_text_embs, self.temperature) # (B, B)
        labels = torch.arange(logits.size(0), device=self.device)
        prototype_text_loss = (self.ce_loss(logits, labels) + self.ce_loss(logits.t(), labels)) / 2.0

        return (prototype_image_loss + prototype_text_loss) / 2.0 

    def _alignment_loss(self, e1, e2):
        '''
            input:
                e1, e2: the embedding with the shape of (B, N_z)
            output:
                contrastive alignment loss
        '''
        logits = self._cos_sim_matrix(e1, e2, self.temperature) # (B, B)
        labels = torch.arange(logits.size(0), device=self.device)
        return (self.ce_loss(logits, labels) + self.ce_loss(logits.t(), labels)) / 2.0
        
    def training_step(self, batch):
        images = batch['image'] # tensor, (B, H, W)
        text_querys = batch['text_query_tokenized'] # already tokenized (B, N_token_dim)
        entity_summarys = batch['entity_summary']  # already tokenized (B, N_token_dim)
        entity_images = batch['entity_image'] # tensor, (B, H, W)
        entity_image_mask = batch['has_entity_img'] # tensor (B, )
        entity_ix = batch['entity_ix'] # tensor (B,)
        triplets_ix = batch['triplets_ix'] # tensor long, (B, N_nt_num+1, N_pt, 3)
        rels_text = batch['rels_text_tokenized'] # tokenized, (B, N_pt, N_token_dim)
        B, N_pt, _ = rels_text.size()
        rels_text_flatten = torch.flatten(rels_text, 0, 1) # (B * N_pt, N_token_dim)
        masks = batch['masks'] # (B, N_nt_num+1, N_pt)
        ## expand masks and rels_text!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        
        # rels_text = rels_text.unsqueeze(1).expand(-1, triplets_ix.size(1), -1, -1) # (B, N_nt_num+1, N_pt, N_token_dim)
        # masks = masks.unsqueeze(1).expand(-1, triplets_ix.size(1), -1) # (B, N_nt_num+1, N_pt)
        ans_embs = self.clip_encoder(images, text_querys) # (B, N_z)
        entity_embs = self.kge.get_node_embs(entity_ix) # (B, N_z)

        entity_image_embs = self.clip_encoder.encode_image(entity_images) # (B, N_z)
        entity_text_embs = self.clip_encoder.encode_text(entity_summarys) # (B, N_z)
        entity_image_embs = entity_image_embs * entity_image_mask[:, None] + entity_text_embs * (1.0 - entity_image_mask[:, None]) # replace image emb. of the entities without sample image with its text emb

        # calculate the knowledge graph embedding loss
        h, r, t= self.kge(triplets_ix) #(B, N_nt_num+1, N_pt, N_z)
        # r = self.clip_encoder.encode_text(rels_text_flatten) #(B * N_pt, N_z)
        # r = torch.unflatten(r, dim=0, sizes=[B, N_pt]) # (B, N_pt, N_z)
        # r = r.unsqueeze(1).expand(-1, triplets_ix.size(1), -1, -1) # (B, N_nt_num+1, N_pt, N_z)
        
        # get global tensors
        if torch.cuda.device_count() > 1:
            h = torch.flatten(self.all_gather(h, sync_grads=True), 0, 1)
            r = torch.flatten(self.all_gather(r, sync_grads=True), 0, 1)
            t = torch.flatten(self.all_gather(t, sync_grads=True), 0, 1)
            masks = torch.flatten(self.all_gather(masks, sync_grads=True), 0, 1)
            ans_embs =  torch.flatten(self.all_gather(ans_embs, sync_grads=True), 0, 1)
            entity_embs = torch.flatten(self.all_gather(entity_embs, sync_grads=True), 0, 1)
            entity_image_embs = torch.flatten(self.all_gather(entity_image_embs, sync_grads=True), 0, 1)
            entity_text_embs = torch.flatten(self.all_gather(entity_text_embs, sync_grads=True), 0, 1)
        # if self.trainer.global_rank == 0:
        #     pdb.set_trace()
        # self.trainer.strategy.barrier()
        ke_loss = self._ke_loss(h,r,t, masks)
        alignment_loss = self._alignment_loss(ans_embs, entity_embs)

        # calculate the prototype alignment loss (pa_loss)
        pa_loss = self._prototype_alignment_loss(entity_embs, entity_image_embs, entity_text_embs)

        self.log('train/alignment_loss', alignment_loss, sync_dist=True)
        self.log('train/ke_loss', ke_loss, sync_dist=True)
        self.log('train/pa_loss', pa_loss, sync_dist=True)

        total_loss = alignment_loss + self.beta1 * ke_loss + self.beta2 *  pa_loss
        self.log('train/total_loss', total_loss, sync_dist=True)
        return total_loss
    
    # def training_step_end(self, batch_parts_outputs):

    #     h = torch.cat(batch_parts_outputs['h'], dim=0)
    #     r = torch.cat(batch_parts_outputs['r'], dim=0)
    #     t = torch.cat(batch_parts_outputs['t'], dim=0)
    #     masks = torch.cat(batch_parts_outputs['masks'], dim=0)
    #     ans_embs = torch.cat(batch_parts_outputs['ans_embs'], dim=0)
    #     entity_embs = torch.cat(batch_parts_outputs['entity_embs'], dim=0)

    #     ke_loss = self._ke_loss(h,r,t, masks)

    #     # calculate the prototype alignment loss (pa_loss)
        
    #     # entity_image_embs = self.clip_encoder.encode_image(entity_images) # (B, N_z)
    #     # entity_text_embs = self.clip_encoder.encode_text(entity_summarys)
    #     # pa_loss = self._prototype_alignment_loss(entity_embs, entity_image_embs, entity_text_embs)

    #     # calculate alignment loss (align_loss)
        
    #     alignment_loss = self._alignment_loss(ans_embs, entity_embs)

    #     return alignment_loss + self.beta1 * ke_loss 

    def validation_step(self, batch):
        images = batch['image'] # tensor, (B, H, W)
        text_querys = batch['text_query_tokenized'] # already tokenized (B, N_token_dim)
        entity_summarys = batch['entity_summary']  # already tokenized (B, N_token_dim)
        entity_images = batch['entity_image'] # tensor, (B, H, W)
        entity_image_mask = batch['has_entity_img'] # tensor (B,)
        entity_ix = batch['entity_ix'] # tensor (B,)

        ans_embs = self.clip_encoder(images, text_querys) # (B, N_z)
        entity_image_embs = self.clip_encoder.encode_image(entity_images) # (B, N_z)
        entity_text_embs = self.clip_encoder.encode_text(entity_summarys) # (B, N_z)
        entity_image_embs = entity_image_embs * entity_image_mask[:, None] + entity_text_embs * (1.0 - entity_image_mask[:, None]) # replace image emb. of the entities without sample image with its text emb

        if torch.cuda.device_count() > 1:
            ans_embs =  torch.flatten(self.all_gather(ans_embs, sync_grads=True), 0, 1)
            entity_image_embs = torch.flatten(self.all_gather(entity_image_embs, sync_grads=True), 0, 1)
            entity_text_embs = torch.flatten(self.all_gather(entity_text_embs, sync_grads=True), 0, 1)


        alignment_loss = self._alignment_loss(ans_embs, entity_text_embs + entity_image_embs)
        self.log('val/alignment_loss', alignment_loss, sync_dist=True)
        return alignment_loss
    
    def on_test_start(self):
        self.entity_embs = []
        entity_summaries = self.kg.get_ents_text(self.kg.entities)
        entity_imgs_path = self.kg.get_ents_image(self.kg.entities)       
        tokenizer = open_clip.get_tokenizer(self.encoder_cfg.model)
        for i in tqdm(range(self.kg.n_ent)):
            with torch.no_grad():
                entity_summary_tokenized = tokenizer(entity_summaries[i]).squeeze().to(self.device)
                entity_image_path = entity_imgs_path[i]
                entity_image, has_entity_img = _load_image_from_path(entity_image_path)
                entity_image = _transform(N_px)(entity_image).to(self.device)
                
                entity_image = entity_image[None, ...]
                entity_summary_tokenized = entity_summary_tokenized[None, ...]
                entity_image_emb = self.clip_encoder.encode_image(entity_image) # (1, N_z)
                entity_text_emb = self.clip_encoder.encode_text(entity_summary_tokenized) # (1, N_z)
                entity_image_emb = entity_image_emb * has_entity_img + entity_text_emb * (1.0 - has_entity_img) # replace image emb. of the entities without sample image with its text emb

                entity_emb = entity_text_emb + entity_image_emb
                self.entity_embs.append(entity_emb)

        self.entity_embs = torch.cat(tuple(self.entity_embs), 0).to(self.device)


    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # entity_ix = self.inference2(batch, self.entity_embs) # (B, )
        entity_ix = self.inference2(batch, self.entity_embs) # (B, )
        for i in range(entity_ix.shape[0]):
            self.test_outputs[dataloader_idx].append({
                "data_id": batch['data_id'][i], 
                "pred_entity_id": self.kg.entities[entity_ix[i]], 
                "entity_id": batch['entity_id'][i],
                "data_split": batch['data_split'][i]
                })
        # for i in range(entity_ix.shape[0]):
        #     output_query.append({
        #         "data_id": batch['data_id'][i], 
        #         "pred_entity_id": self.kg.entities[entity_ix[i]], 
        #         "entity_id": batch['entity_id'][i],
        #         "data_split": batch['data_split'][i]
        #         })

    def on_test_epoch_end(self):
        """
        `outputs` is a list containing outputs for each test dataset.
        Example structure:
        outputs = [
            [batch_output1, batch_output2, ...],  # Test dataset entity
            [batch_output1, batch_output2, ...]   # Test dataset query
        ]
        """
        output_entity_all = self.test_outputs[0]
        output_query_all = self.test_outputs[1]
        final_result = evaluate_oven_full(output_query_all, output_entity_all)
        self.log('test/query_score', final_result['query_score'], add_dataloader_idx=False)
        self.log('test/entity_score', final_result['entity_score'], add_dataloader_idx=False)
        self.log('test/final_score', final_result['final_score'], add_dataloader_idx=False)
        self.log('test/query_seen_result', final_result['query_result']['seen'], add_dataloader_idx=False)
        self.log('test/query_unseen_result', final_result['query_result']['unseen'], add_dataloader_idx=False)
        self.log('test/entity_seen_result', final_result['entity_result']['seen'], add_dataloader_idx=False)
        self.log('test/entity_unseen_result', final_result['entity_result']['unseen'], add_dataloader_idx=False)

    def inference(self, batch):
        images = batch['image'].to(self.device) # tensor, (B, H, W)
        text_querys = batch['text_query_tokenized'].to(self.device) # already tokenized (B, N_token_dim)
        ans_embs = self.clip_encoder(images, text_querys) # (B, N_z)

        ans_embs = ans_embs[:, None, :] # (B, 1, N_z)
        all_node_embs = self.kge.get_all_node_embs() # tensor (n_ent, N_z)
        cos_sim = self._cos_sim(ans_embs, all_node_embs, t=self.temperature) # tensor (B, n_ent)
        entity_ix = torch.argmax(cos_sim, dim=-1) # tensor (B, )
        return entity_ix

    def inference2(self, batch, entity_embs):
        images = batch['image'].to(self.device) # tensor, (B, H, W)
        text_querys = batch['text_query_tokenized'].to(self.device) # already tokenized (B, N_token_dim)
        ans_embs = self.clip_encoder(images, text_querys) # (B, N_z)

        ans_embs = ans_embs[:, None, :] # (B, 1, N_z)
        entity_embs = entity_embs[None, ...] #(1, N_ent, N_z)
        cos_sim = self._cos_sim(ans_embs, entity_embs, t=self.temperature) # tensor (B, n_ent)
        entity_ix = torch.argmax(cos_sim, dim=-1) # tensor (B, )
        return entity_ix

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer, params=self.parameters())
        if self.lr_scheduler:
            lr_scheduler = hydra.utils.instantiate(self.lr_scheduler, optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': lr_scheduler
            }
        else:
            return optimizer