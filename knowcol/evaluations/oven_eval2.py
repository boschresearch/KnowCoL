import os
import torch
import open_clip
from evaluation import evaluate_oven_full
from tqdm import tqdm
from knowcol.models.knowcol import KnowCoL as KnowCoL
from knowcol.models.knowcol4 import KnowCoL as KnowCoL4
from knowcol.datasets.datasets import OvenTestDataset
from torchvision.transforms import v2
from PIL import Image
from torch.utils.data import DataLoader
from knowcol.datasets.kg import KG
import json
from pytorch_lightning import Callback, LightningModule, Trainer
from knowcol.utils.utils import get_last_checkpoint
from pathlib import Path
import knowcol
from knowcol.datasets.utils import _load_image_from_path
import json

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
        return image.convert("RGB")

def to_absolute_path(path: str):
    path = Path(path)
    if not path.is_absolute():
        path = Path(knowcol.__file__).parent.parent / path
    return path

def _transform(n_px):
    return v2.Compose([
        v2.Resize(n_px, interpolation=BICUBIC),
        v2.CenterCrop(n_px),
        _convert_image_to_rgb,
        v2.ToTensor(),
        v2.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


knowledge_base_path='dataset/knowledge_base'
entity_set_path='dataset/wikidata5m_s/entity.txt'
relation_path='dataset/wikidata5m_s/relation.txt'
triplet_h_path='dataset/wikidata5m_s/triplet_h.jsonl'
triplet_t_path='dataset/wikidata5m_s/triplet_t.jsonl'

########################## Things need to be changed ##############################
data_dir = 'dataset'
jsonl_entity_file = ['oven_entity_test.jsonl']
jsonl_query_file = ['oven_query_test.jsonl']
chk_dir = '/home/zho1rng/oven_train/checkpoints/2025-06-03/16-14-59/saved_models/epoch=2.ckpt'
model = 'KnowCoL4'
device = 'cuda' # cuda
n_px = 224
tokenizer = 'ViT-L-14'
########################## Things need to be changed ##############################




kg = KG(
        knowledge_base_path=knowledge_base_path, 
        entity_set_path=entity_set_path, 
        relation_path=relation_path,
        triplet_h_path=triplet_h_path,
        triplet_t_path=triplet_t_path
        )

model = eval(model).load_from_checkpoint(chk_dir)
tokenizer = open_clip.get_tokenizer(tokenizer)

test_dataset_entity = OvenTestDataset(data_dir, 'test_data', jsonl_entity_file, kg, _transform(n_px=n_px), tokenizer=tokenizer)
test_loader_entity = DataLoader(test_dataset_entity, batch_size=128, num_workers=4)

test_dataset_query = OvenTestDataset(data_dir, 'test_data', jsonl_query_file, kg, _transform(n_px=n_px), tokenizer=tokenizer)
test_loader_query = DataLoader(test_dataset_query, batch_size=128, num_workers=4)

model.to(device)
model.eval()

entity_embs = []
entity_summaries = kg.get_ents_text(kg.entities)
entity_imgs_path = kg.get_ents_image(kg.entities)       
for i in tqdm(range(kg.n_ent)):
    with torch.no_grad():
        entity_summary_tokenized = tokenizer(entity_summaries[i]).squeeze().to(device)
        entity_image_path = entity_imgs_path[i]
        entity_image, has_entity_img = _load_image_from_path(entity_image_path)
        entity_image = _transform(n_px)(entity_image).to(device)
        
        entity_image = entity_image[None, ...]
        entity_summary_tokenized = entity_summary_tokenized[None, ...]
        entity_image_emb = model.clip_encoder.encode_image(entity_image) # (B, N_z)
        entity_text_emb = model.clip_encoder.encode_text(entity_summary_tokenized) # (B, N_z)
        entity_image_emb = entity_image_emb * has_entity_img + entity_text_emb * (1.0 - has_entity_img) # replace image emb. of the entities without sample image with its text emb

        entity_emb = entity_text_emb + entity_image_emb
        entity_embs.append(entity_emb)

entity_embs = torch.cat(tuple(entity_embs), 0).to(device)


output_entity = []
output_query = []

save_path = Path(chk_dir).expanduser().parent.parent

with torch.no_grad():
    for batch in tqdm(test_loader_entity):
        entity_ix = model.inference2(batch, entity_embs) # (B, )
        for i in range(entity_ix.shape[0]):
             output_entity.append({
                  "data_id": batch['data_id'][i], 
                  "pred_entity_id": kg.entities[entity_ix[i]], 
                  "entity_id": batch['entity_id'][i],
                  "data_split": batch['data_split'][i]
                  })
    # Writing list to JSONL
    with open(save_path / "entity.jsonl", 'w') as file:
        for entry in output_entity:
            json_line = json.dumps(entry)  # Convert dictionary to JSON string
            file.write(json_line + '\n')  # Write the JSON string followed by a newline
    
    for batch in tqdm(test_loader_query):
        entity_ix = model.inference2(batch, entity_embs) # (B, )
        for i in range(entity_ix.shape[0]):
             output_query.append({
                  "data_id": batch['data_id'][i], 
                  "pred_entity_id": kg.entities[entity_ix[i]], 
                  "entity_id": batch['entity_id'][i],
                  "data_split": batch['data_split'][i]
                  })

    with open(save_path / "query.jsonl", 'w') as file:
        for entry in output_query:
            json_line = json.dumps(entry)  # Convert dictionary to JSON string
            file.write(json_line + '\n')  # Write the JSON string followed by a newline
    

final_results = evaluate_oven_full(output_query, output_entity)


with open(save_path / "results.txt", "w") as file:
    json.dump(final_results, file, indent=4)

