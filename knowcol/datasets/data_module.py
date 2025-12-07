import logging
import os
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from PIL import Image
Image.MAX_IMAGE_PIXELS = 200000000
import knowcol
from .datasets import OvenDataset
from .kg import KG

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import open_clip

def _convert_image_to_rgb(image):
        return image.convert("RGB")

def _transform(n_px):
    return v2.Compose([
        v2.Resize(n_px, interpolation=BICUBIC),
        v2.CenterCrop(n_px),
        _convert_image_to_rgb,
        v2.ToTensor(),
        v2.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset_cfg: DictConfig,
        val_dataset_cfg: DictConfig,
        num_workers: int,
        batch_size: int,
        tokenizer: str, 
        n_px: int,
        shuffle_val:str, 
        kg: KG,
        test_entity_dataset_cfg: DictConfig = None, 
        test_query_dataset_cfg: DictConfig = None
    ):
        super().__init__()
        self.train_dataset_cfg = train_dataset_cfg
        self.val_dataset_cfg = val_dataset_cfg
        self.test_entity_dataset_cfg = test_entity_dataset_cfg
        self.test_query_dataset_cfg = test_query_dataset_cfg
        self.num_workers = num_workers
        self.shuffle_val = shuffle_val
        self.batch_size = batch_size
        self.n_px = n_px
        self.tokenizer = tokenizer
        self.kg = kg
        self._setup()


    def _setup(self, stage=None):
        transforms = _transform(self.n_px)
        tokenizer = open_clip.get_tokenizer(self.tokenizer)
        self.train_dataset = hydra.utils.instantiate(self.train_dataset_cfg, kg=self.kg, transform=transforms, tokenizer=tokenizer)
        self.val_dataset = hydra.utils.instantiate(self.val_dataset_cfg, kg=self.kg, transform=transforms, tokenizer=tokenizer)
        if self.test_entity_dataset_cfg:
            self.test_entity_dataset = hydra.utils.instantiate(self.test_entity_dataset_cfg, kg=self.kg, transform=transforms, tokenizer=tokenizer)
        if self.test_query_dataset_cfg:    
            self.test_query_dataset = hydra.utils.instantiate(self.test_query_dataset_cfg, kg=self.kg, transform=transforms, tokenizer=tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=self.shuffle_val, num_workers=self.num_workers)
    
    def test_dataloader(self):
        if self.test_entity_dataset and self.test_query_dataset:
            return [
                DataLoader(self.test_entity_dataset, batch_size=128, shuffle=self.shuffle_val, num_workers=self.num_workers),
                DataLoader(self.test_query_dataset, batch_size=128, shuffle=self.shuffle_val, num_workers=self.num_workers)
            ]
        elif self.test_entity_dataset:
            return DataLoader(self.test_entity_dataset, batch_size=128, shuffle=self.shuffle_val, num_workers=self.num_workers)
        elif self.test_query_dataset_cfg:
            return DataLoader(self.test_query_dataset, batch_size=128, shuffle=self.shuffle_val, num_workers=self.num_workers)
        else:
            return super().test_dataloader()