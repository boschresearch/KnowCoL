from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
import json
from knowcol.datasets.utils import pil_loader, get_blanc_img
import logging
from tqdm import tqdm
from omegaconf import DictConfig
import hydra
import random
from .triplet import Triplet
import open_clip
from .kg import KG
import knowcol
import pdb
import os
logger = logging.getLogger(__name__)

extensions = ['.JPEG', '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

class OvenDataset(Dataset):
    def __init__(
            self, 
            data_dir: str, 
            jsonl_files: List[str], 
            max_pt_num:int, 
            negative_sampling_num: int, 
            kg: KG, 
            transform: Callable,
            tokenizer: Callable,
            query_repeat: int = 20
            ):
        '''
        Arguments:
            data_dir (string): the data dir path
            jsonl_file (string): json file name which include necessary information
            transform (callable, optional): Optional transform to be applied on a sample.
        '''
        data_dir = Path(data_dir)
        if not data_dir.is_absolute():
            data_dir = Path(knowcol.__file__).parent.parent / data_dir
        self.data_dir = data_dir

        self.transform = transform
        self.tokenizer = tokenizer
        self.frame = []
        self.frame_len = 0
        self.max_pt_num = max_pt_num
        self.negative_sampling_num = negative_sampling_num
        self.kg = kg
        self.n_ent = self.kg.n_ent
        self.query_repeat = query_repeat
        self._read_jsonl(jsonl_files)

    def _read_jsonl(self, jsonl_files):
        for file in jsonl_files:
            with open(self.data_dir / 'oven_data' / file) as f:
                for line in f:
                    json_object = json.loads(line)
                    if 'query' in file:
                        for _ in range(self.query_repeat):
                            self.frame.append(json_object)
                    else:
                        self.frame.append(json_object)
        
        self.frame_len = len(self.frame)

    def imgid2path(self, id: str, ext: str):
        if id.startswith('oven_00'):
            return self.data_dir / 'oven_images' / '00' / (id + ext)
        elif id.startswith('oven_01'):
            return self.data_dir / 'oven_images' / '01' / (id + ext)
        elif id.startswith('oven_02'):
            return self.data_dir / 'oven_images' / '02' / (id + ext)
        elif id.startswith('oven_03'):
            return self.data_dir / 'oven_images' / '03' / (id + ext)
        elif id.startswith('oven_04'):
            return self.data_dir / 'oven_images' / '04' / (id + ext)
        elif id.startswith('oven_05'):
            return self.data_dir / 'oven_images' / '05' / (id + ext)
    
    @staticmethod
    def random_int(max, ext):
        r = random.randint(1, max)
        while r == ext:
            r = random.randint(1, max)
        return r

    def randomly_corrupt_triplets(self, triplets: torch.Tensor, directions: List):
        corrupt_triplets = triplets.clone() # (N, 3)
        for i in range(corrupt_triplets.size(0)):
            if directions[i] == 'out':
                if corrupt_triplets[i][2]!= 0: # 0 is the pad index
                    corrupt_triplets[i][2] = self.random_int(self.n_ent, corrupt_triplets[i][2])
            else:
                if corrupt_triplets[i][0]!= 0: # 0 is the pad index
                    corrupt_triplets[i][0] = self.random_int(self.n_ent, corrupt_triplets[i][0])
        
        return corrupt_triplets

    
    def pad_or_crop(self, triplets, max_num):
        '''
            input:
                triplets: a list of triplet
                max_num: max number of triplets
            output:
                a list of padded or cropped triplets
                mask: tensor with the shape of (max_num,)
        '''
        if len(triplets) > max_num:
            mask = torch.ones(max_num)
            return random.sample(triplets, max_num), mask
        else:
            zeros = [Triplet((0,0,0), 'none', 'to')] * (max_num - len(triplets))
            mask = torch.cat((torch.ones(len(triplets)), torch.zeros(max_num - len(triplets))), dim=0)
            return triplets + zeros, mask

    def _get_rel_text(self, pid:str) -> str:
        return self.kg.get_rel_text(pid)
        
    def get_rels_text(self, tripplets_ix):
        return list(map(lambda t: self._get_rel_text(t[1])))

    def _load_image(self, img_id):
        for ext in extensions:
            img_path = self.imgid2path(img_id, ext)
            if os.path.exists(img_path):
                return pil_loader(img_path)
        
        assert False, f"No image with this id {img_id}!"
    
    def _load_image_from_path(self, img_path):
        try:
            return pil_loader(img_path), 1
        except Exception as e:
            return get_blanc_img(), 0


    def __len__(self):
        return self.frame_len
    
    def __getitem__(self, index: int) ->  Tuple[Any, Any, Any]:
        data = self.frame[index]
        data_id = data['data_id']
        image = self._load_image(data['image_id'])
        if self.transform:
            image = self.transform(image)
        question = data['question']
        question_tokenized = self.tokenizer(question).squeeze() # (N_token_dim)

        entity_id = data['entity_id']
        entity_ix = self.kg.ent2ix[entity_id]

        entity_summary = self.kg.get_ent_text(entity_id)
        entity_summary_tokenized = self.tokenizer(entity_summary).squeeze()
        entity_image_path = self.kg.get_ent_image(entity_id) # return the path
        entity_image, has_entity_img = self._load_image_from_path(entity_image_path) # some entities do not have sample image
        if self.transform:
            entity_image = self.transform(entity_image)

        triplets_h = self.kg.get_triplets_h(entity_id) #  A list of triplets
        triplets_t = self.kg.get_triplets_t(entity_id) #  A list of triplets
        # triplets_h_ix = torch.tensor(list(map(lambda t: t.ix_form, triplets_h)), dtype=torch.long)
        # triplets_t_ix = torch.tensor(list(map(lambda t: t.ix_form, triplets_t)), dtype=torch.long)


        triplets, mask = self.pad_or_crop(triplets_h + triplets_t, self.max_pt_num) # pad or crop the triplets

        triplets_ix = torch.tensor(list(map(lambda t: t.ix_form, triplets)), dtype=torch.long) # (N_max_num, 3) 
        rels_text_tokenized = self.tokenizer(list(map(lambda t: t.rel_text, triplets))) # (N_max_num, N_token_dim)
        directions = list(map(lambda t: t.direction, triplets))
        triplets_ix_all = [triplets_ix]
        for i in range(self.negative_sampling_num):
            t = self.randomly_corrupt_triplets(triplets_ix, directions)
            triplets_ix_all.append(t)
        triplets_ix_all = torch.stack(triplets_ix_all, dim=0) # tensor, (negative_sampling_num+1, N_max_num, 3)
        # rels_text_tokenized = rels_text_tokenized.unsqueeze(0).expand(self.negative_sampling_num+1, -1, -1) # tensor, (negative_sampling_num+1, N_pt, N_token_dim)
        # mask = mask.unsqueeze(0).expand(self.negative_sampling_num+1, -1) # (negative_sampling_num+1, N_max_num)
        ret =  {
            'image': image,
            'text_query_tokenized': question_tokenized,
            'entity_ix': entity_ix,
            'entity_summary': entity_summary_tokenized,
            'entity_image': entity_image,
            'has_entity_img': has_entity_img,
            'triplets_ix': triplets_ix_all,
            'rels_text_tokenized': rels_text_tokenized,
            'masks': mask
        }
        return ret
    
## without multiple query dataset
class OvenDataset2(Dataset):
    def __init__(
            self, 
            data_dir: str, 
            jsonl_files: List[str], 
            max_pt_num:int, 
            negative_sampling_num: int, 
            kg: KG, 
            transform: Callable,
            tokenizer: Callable,
            query_repeat: int = 20
            ):
        '''
        Arguments:
            data_dir (string): the data dir path
            jsonl_file (string): json file name which include necessary information
            transform (callable, optional): Optional transform to be applied on a sample.
        '''
        data_dir = Path(data_dir)
        if not data_dir.is_absolute():
            data_dir = Path(knowcol.__file__).parent.parent / data_dir
        self.data_dir = data_dir

        self.transform = transform
        self.tokenizer = tokenizer
        self.frame = []
        self.frame_len = 0
        self.max_pt_num = max_pt_num
        self.negative_sampling_num = negative_sampling_num
        self.kg = kg
        self.n_ent = self.kg.n_ent
        self.query_repeat = query_repeat
        self._read_jsonl(jsonl_files)

    def _read_jsonl(self, jsonl_files):
        for file in jsonl_files:
            with open(self.data_dir / 'oven_data' / file) as f:
                for line in f:
                    json_object = json.loads(line)
                    self.frame.append(json_object)
        
        self.frame_len = len(self.frame)

    def imgid2path(self, id: str, ext: str):
        if id.startswith('oven_00'):
            return self.data_dir / 'oven_images' / '00' / (id + ext)
        elif id.startswith('oven_01'):
            return self.data_dir / 'oven_images' / '01' / (id + ext)
        elif id.startswith('oven_02'):
            return self.data_dir / 'oven_images' / '02' / (id + ext)
        elif id.startswith('oven_03'):
            return self.data_dir / 'oven_images' / '03' / (id + ext)
        elif id.startswith('oven_04'):
            return self.data_dir / 'oven_images' / '04' / (id + ext)
        elif id.startswith('oven_05'):
            return self.data_dir / 'oven_images' / '05' / (id + ext)
    
    @staticmethod
    def random_int(max, ext):
        r = random.randint(1, max)
        while r == ext:
            r = random.randint(1, max)
        return r

    def randomly_corrupt_triplets(self, triplets: torch.Tensor, directions: List):
        corrupt_triplets = triplets.clone() # (N, 3)
        for i in range(corrupt_triplets.size(0)):
            if directions[i] == 'out':
                if corrupt_triplets[i][2]!= 0: # 0 is the pad index
                    corrupt_triplets[i][2] = self.random_int(self.n_ent, corrupt_triplets[i][2])
            else:
                if corrupt_triplets[i][0]!= 0: # 0 is the pad index
                    corrupt_triplets[i][0] = self.random_int(self.n_ent, corrupt_triplets[i][0])
        
        return corrupt_triplets

    
    def pad_or_crop(self, triplets, max_num):
        '''
            input:
                triplets: a list of triplet
                max_num: max number of triplets
            output:
                a list of padded or cropped triplets
                mask: tensor with the shape of (max_num,)
        '''
        if len(triplets) > max_num:
            mask = torch.ones(max_num)
            return random.sample(triplets, max_num), mask
        else:
            zeros = [Triplet((0,0,0), 'none', 'to')] * (max_num - len(triplets))
            mask = torch.cat((torch.ones(len(triplets)), torch.zeros(max_num - len(triplets))), dim=0)
            return triplets + zeros, mask

    def _get_rel_text(self, pid:str) -> str:
        return self.kg.get_rel_text(pid)
        
    def get_rels_text(self, tripplets_ix):
        return list(map(lambda t: self._get_rel_text(t[1])))

    def _load_image(self, img_id):
        for ext in extensions:
            img_path = self.imgid2path(img_id, ext)
            if os.path.exists(img_path):
                return pil_loader(img_path)
        
        assert False, f"No image with this id {img_id}!"
    
    def _load_image_from_path(self, img_path):
        try:
            return pil_loader(img_path), 1
        except Exception as e:
            return get_blanc_img(), 0


    def __len__(self):
        return self.frame_len
    
    def __getitem__(self, index: int) ->  Tuple[Any, Any, Any]:
        data = self.frame[index]
        data_id = data['data_id']
        image = self._load_image(data['image_id'])
        if self.transform:
            image = self.transform(image)
        question = data['question']
        question_tokenized = self.tokenizer(question).squeeze() # (N_token_dim)

        entity_id = data['entity_id']
        entity_ix = self.kg.ent2ix[entity_id]

        entity_summary = self.kg.get_ent_text(entity_id)
        entity_summary_tokenized = self.tokenizer(entity_summary).squeeze()
        entity_image_path = self.kg.get_ent_image(entity_id) # return the path
        entity_image, has_entity_img = self._load_image_from_path(entity_image_path) # some entities do not have sample image
        if self.transform:
            entity_image = self.transform(entity_image)

        triplets_h = self.kg.get_triplets_h(entity_id) #  A list of triplets
        triplets_t = self.kg.get_triplets_t(entity_id) #  A list of triplets
        # triplets_h_ix = torch.tensor(list(map(lambda t: t.ix_form, triplets_h)), dtype=torch.long)
        # triplets_t_ix = torch.tensor(list(map(lambda t: t.ix_form, triplets_t)), dtype=torch.long)


        triplets, mask = self.pad_or_crop(triplets_h + triplets_t, self.max_pt_num) # pad or crop the triplets

        triplets_ix = torch.tensor(list(map(lambda t: t.ix_form, triplets)), dtype=torch.long) # (N_max_num, 3) 
        rels_text_tokenized = self.tokenizer(list(map(lambda t: t.rel_text, triplets))) # (N_max_num, N_token_dim)
        directions = list(map(lambda t: t.direction, triplets))

        triplets_ix_all = [triplets_ix]
        
        for i in range(self.negative_sampling_num):
            t = self.randomly_corrupt_triplets(triplets_ix, directions)
            triplets_ix_all.append(t)
        triplets_ix_all = torch.stack(triplets_ix_all, dim=0) # tensor, (negative_sampling_num+1, N_max_num, 3)
        masks = torch.stack(masks, dim=0) # (negative_sampling_num+1, N_max_num)
        # rels_text_tokenized = rels_text_tokenized.unsqueeze(0).expand(self.negative_sampling_num+1, -1, -1) # tensor, (negative_sampling_num+1, N_pt, N_token_dim)
        # mask = mask.unsqueeze(0).expand(self.negative_sampling_num+1, -1) # (negative_sampling_num+1, N_max_num)
        ret =  {
            'image': image,
            'text_query_tokenized': question_tokenized,
            'entity_ix': entity_ix,
            'entity_summary': entity_summary_tokenized,
            'entity_image': entity_image,
            'has_entity_img': has_entity_img,
            'triplets_ix': triplets_ix_all,
            'rels_text_tokenized': rels_text_tokenized,
            'masks': mask
        }
        return ret
    

# class OvenTestDataset(Dataset):
#     def __init__(
#         self, 
#         data_dir: str,
#         jsonl_folder: str, 
#         entity_jsonl_files: List[str],
#         query_jsonl_files: List[str], 
#         kg: KG, 
#         transform: Callable,
#         tokenizer: Callable
#         ):
#         data_dir = Path(data_dir)
#         if not data_dir.is_absolute():
#             data_dir = Path(knowcol.__file__).parent.parent / data_dir
#         self.data_dir = data_dir

#         self.transform = transform
#         self.tokenizer = tokenizer
#         self.ent_frame = []
#         self.query_frame = []
#         self.frame_ent_len = 0
#         self.frame_query_len = 0
#         self.kg = kg
#         self.n_ent = self.kg.n_ent
#         self.jsonl_folder = jsonl_folder
#         self._read_jsonl(entity_jsonl_files, query_jsonl_files)

#     def _read_jsonl(self, entity_jsonl_files, query_jsonl_files):
#         for file in entity_jsonl_files:
#             with open(self.data_dir / self.jsonl_folder/ file) as f:
#                 for line in f:
#                     json_object = json.loads(line)
#                     self.ent_frame.append(json_object)
        
#         for file in query_jsonl_files:
#             with open(self.data_dir / self.jsonl_folder/ file) as f:
#                 for line in f:
#                     json_object = json.loads(line)
#                     self.query_frame.append(json_object)

#         self.frame_ent_len = len(self.ent_frame)
#         self.frame_query_len = len(self.query_frame)
    
#     def imgid2path(self, id: str, ext: str):
#         if id.startswith('oven_00'):
#             return self.data_dir / 'oven_images' / '00' / (id + ext)
#         elif id.startswith('oven_01'):
#             return self.data_dir / 'oven_images' / '01' / (id + ext)
#         elif id.startswith('oven_02'):
#             return self.data_dir / 'oven_images' / '02' / (id + ext)
#         elif id.startswith('oven_03'):
#             return self.data_dir / 'oven_images' / '03' / (id + ext)
#         elif id.startswith('oven_04'):
#             return self.data_dir / 'oven_images' / '04' / (id + ext)
#         elif id.startswith('oven_05'):
#             return self.data_dir / 'oven_images' / '05' / (id + ext)
        
#     def _get_rel_text(self, pid:str) -> str:
#         return self.kg.get_rel_text(pid)
        
#     def get_rels_text(self, tripplets_ix):
#         return list(map(lambda t: self._get_rel_text(t[1])))

#     def _load_image(self, img_id):
#         for ext in extensions:
#             try:
#                 img_path = self.imgid2path(img_id, ext)
#                 return pil_loader(img_path)
#             except Exception as e:
#                 pass
        
#         assert('No image with this id!')


class OvenTestDataset(Dataset):
    def __init__(
            self, 
            data_dir: str, 
            jsonl_folder: str, 
            jsonl_files: List[str], 
            kg: KG, 
            transform: Callable,
            tokenizer: Callable
            ):
        '''
        Arguments:
            data_dir (string): the data dir path
            jsonl_file (string): json file name which include necessary information
            transform (callable, optional): Optional transform to be applied on a sample.
        '''
        data_dir = Path(data_dir)
        if not data_dir.is_absolute():
            data_dir = Path(knowcol.__file__).parent.parent / data_dir
        self.data_dir = data_dir

        self.transform = transform
        self.tokenizer = tokenizer
        self.frame = []
        self.frame_len = 0
        self.kg = kg
        self.n_ent = self.kg.n_ent
        self.jsonl_folder = jsonl_folder
        self._read_jsonl(jsonl_files)


    def _read_jsonl(self, jsonl_files):
        for file in jsonl_files:
            with open(self.data_dir / self.jsonl_folder/ file) as f:
                for line in f:
                    json_object = json.loads(line)
                    self.frame.append(json_object)
        
        self.frame_len = len(self.frame)

    def imgid2path(self, id: str, ext: str):
        if id.startswith('oven_00'):
            return self.data_dir / 'oven_images' / '00' / (id + ext)
        elif id.startswith('oven_01'):
            return self.data_dir / 'oven_images' / '01' / (id + ext)
        elif id.startswith('oven_02'):
            return self.data_dir / 'oven_images' / '02' / (id + ext)
        elif id.startswith('oven_03'):
            return self.data_dir / 'oven_images' / '03' / (id + ext)
        elif id.startswith('oven_04'):
            return self.data_dir / 'oven_images' / '04' / (id + ext)
        elif id.startswith('oven_05'):
            return self.data_dir / 'oven_images' / '05' / (id + ext)

    def _get_rel_text(self, pid:str) -> str:
        return self.kg.get_rel_text(pid)
        
    def get_rels_text(self, tripplets_ix):
        return list(map(lambda t: self._get_rel_text(t[1])))

    def _load_image(self, img_id):
        for ext in extensions:
            try:
                img_path = self.imgid2path(img_id, ext)
                return pil_loader(img_path)
            except Exception as e:
                pass
        
        assert('No image with this id!')

    def __len__(self):
        return self.frame_len
    
    def __getitem__(self, index: int) ->  Tuple[Any, Any, Any]:
        data = self.frame[index]
        data_id = data['data_id']
        image = self._load_image(data['image_id'])
        if self.transform:
            image = self.transform(image)
        question = data['question']
        question_tokenized = self.tokenizer(question).squeeze() # (N_token_dim)
        entity_id = data['entity_id']
        data_split = data['data_split']
        ret =  {
            'image': image,
            'text_query_tokenized': question_tokenized,
            'data_id': data_id,
            'entity_id': entity_id,
            'data_split': data_split
        }
        return ret
    
class OvenKnowledgeBaseDataset(Dataset):
    def __init__(
            self, 
            kg: KG, 
            transform: Callable,
            tokenizer: Callable,
            n_px: int = 224
        ):
        self.kg = kg
        self.transform = transform
        self.tokenizer = tokenizer
        self.n_px = n_px
        self.entity_summaries = kg.get_ents_text(kg.entities)
        self.entity_imgs_path = kg.get_ents_image(kg.entities)
        
    def _load_image_from_path(self, img_path):
        try:
            return pil_loader(img_path), 1
        except Exception as e:
            return get_blanc_img(), 0
        
    def __len__(self):
        return len(self.entity_image)
    
    def __getitem__(self, index: int):
        entity_image_path = self.entity_imgs_path[index]
        entity_summary_tokenized = self.tokenizer(self.entity_summaries[index]).squeeze()
        entity_image, has_entity_img = self._load_image_from_path(entity_image_path)
        entity_image = self.transform(self.n_px)(entity_image)
        ret =  {
            'entity_image': entity_image,
            'entity_summary_tokenized': entity_summary_tokenized
        }
        return ret