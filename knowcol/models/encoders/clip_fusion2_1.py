import torch
import torch.nn as nn 
from PIL import Image
import open_clip
import pdb
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)

def text_global_pool(x, text: Optional[torch.Tensor] = None, pool_type: str = 'argmax'):
    if pool_type == 'first':
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == 'last':
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x

    return pooled, tokens

class CLIPWrapper(nn.Module):
    def __init__(self, model: str, 
                 pretrained: str, 
                 row_clip_dim: int,
                 clip_dim: int, 
                 latent_dim: int,
                 fuser_nhead:int = 8,
                 fuser_num_layers:int =2,
                 isFrozen: bool = False
                 ):
        '''
            model: the name of the clip model
            pretrained: the config of pretrained checkpoint
            clip_dim: the out dimension of clip
            latent_dim: the latent dimension of the joint feature space
            isFrozen: if the parameters of pre-trained clip is frozen
        '''
        super(CLIPWrapper, self).__init__()
        model, _, self.preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained)
        self.model = model
        if isFrozen:
            self.model.eval()  # Freeze the model
            self.model.requires_grad_(False) 
        self.mlp0 = nn.Linear(row_clip_dim, latent_dim) # local image tokens
        self.mlp1 = nn.Linear(clip_dim, latent_dim) # local text tokens
        self.mlp2 = nn.Linear(clip_dim, latent_dim) # global image token
        self.mlp3 = nn.Linear(clip_dim, latent_dim) # global text token

        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=fuser_nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=fuser_num_layers)
        # self.tokenizer = open_clip.get_tokenizer(model)
    
    def encode_text_local_global(self, text):
        cast_dtype = self.model.transformer.get_cast_dtype()

        x = self.model.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.model.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x, tokens = text_global_pool(x, text, self.model.text_pool_type)
        if self.model.text_projection is not None:
            if isinstance(self.model.text_projection, nn.Linear):
                x = self.model.text_projection(x)
            else:
                x = x @ self.model.text_projection
        return x, tokens

    def encode_image(self, imgs):
        '''
            imgs: The images to encode 
            return: image embedding(s) of the global meaning
        '''
        return self.mlp2(self.model.encode_image(imgs))
    
    def encode_text(self, texts: torch.Tensor):
        '''
            texts: The texts to encode
            return: text embedding(s)
        '''
        return self.mlp3(self.model.encode_text(texts))

    
    def forward(self, imgs, texts, image_mask=None):
        '''
            imgs: The images to encode
            texts: The text to encode
            mask: if it isn't None, then it indicates if the images need to be masked
        '''
        # x = self.model.visual.conv1(imgs)  # shape = [*, width, grid, grid]
        # x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        # x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # # class embeddings and positional embeddings
        # x = torch.cat([_expand_token(self.model.visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # # shape = [*, grid ** 2 + 1, width]
        # x = x + self.model.visual.positional_embedding.to(x.dtype)

        # x = self.model.visual.patch_dropout(x)
        # x = self.model.visual.ln_pre(x)

        # x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.model.visual.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        self.model.visual.output_tokens = True
        global_image_embs, image_tokens = self.model.visual(imgs)
        self.model.visual.output_tokens = False

        global_text_embs, text_tokens = self.encode_text_local_global(texts)

        global_image_embs = self.mlp2(global_image_embs)[:,None,:] #(B, 1, N_z)
        image_tokens = self.mlp0(image_tokens) #(B, L, N_z)
        global_text_embs = self.mlp3(global_text_embs)[:,None,:] #(B, 1, N_z)
        text_tokens = self.mlp1(text_tokens) #(B, L, N_z)

        cat_emb = torch.cat((global_image_embs, image_tokens, global_text_embs, text_tokens), dim=1) # (B, N_patch+1, N_z)
        cat_emb = cat_emb.permute(1, 0, 2) # (N_patch+1, B, N_z)
        return self.transformer_encoder(cat_emb)[0,:,:] # use the first token embedding as the global embedding

    # def forward(self, imgs, texts):
    #     '''
    #         imgs: The images to encode
    #         texts: The text to encode
    #         mask: if it isn't None, then it indicates if the images need to be masked
    #     '''
    #     img_embs, text_embs = self.encode_image(imgs), self.encode_text(texts) # (B, N_z)
    #     return img_embs + text_embs # use the first token embedding as the global embedding
    

    

