import torch
import torch.nn as nn 
from PIL import Image
import open_clip

class CLIPWrapper(nn.Module):
    def __init__(self, model: str, 
                 pretrained: str, 
                 clip_dim: int, 
                 latent_dim: int,
                 fuser_nhead:int = 8,
                 fuser_num_layers:int =2,
                 isFrozen: bool = False,

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

        self.mlp1 = nn.Linear(clip_dim, latent_dim)
        self.mlp2 = nn.Linear(clip_dim, latent_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=fuser_nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=fuser_num_layers)
        # self.tokenizer = open_clip.get_tokenizer(model)
    
    def encode_image(self, imgs):
        '''
            imgs: The images to encode 
            return: image embedding(s)
        '''
        return self.mlp1(self.model.encode_image(imgs))
    
    def encode_text(self, texts: torch.Tensor):
        '''
            texts: The texts to encode
            return: text embedding(s)
        '''
        return self.mlp2(self.model.encode_text(texts))

    
    def forward(self, imgs, texts, image_mask=None):
        '''
            imgs: The images to encode
            texts: The text to encode
            mask: if it isn't None, then it indicates if the images need to be masked
        '''
        img_embs, text_embs = self.encode_image(imgs), self.encode_text(texts) # (B, N_z)
        if image_mask:
            img_embs = img_embs * image_mask[:, None] + text_embs * (1.0 - image_mask[:, None]) # replace image emb. of the entities without sample image with its text emb
        cat_emb = torch.stack((img_embs, text_embs), dim=1) # (B, 2, N_z)
        cat_emb = cat_emb.permute(1, 0, 2) # (2, B, N_z)
        return torch.mean(self.transformer_encoder(cat_emb), dim=0) # use the mean embedding as the global embedding

    # def forward(self, imgs, texts):
    #     '''
    #         imgs: The images to encode
    #         texts: The text to encode
    #         mask: if it isn't None, then it indicates if the images need to be masked
    #     '''
    #     img_embs, text_embs = self.encode_image(imgs), self.encode_text(texts) # (B, N_z)
    #     return img_embs + text_embs # use the first token embedding as the global embedding
    

    

