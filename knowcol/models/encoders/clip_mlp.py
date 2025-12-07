import torch
import torch.nn as nn 
from PIL import Image
import open_clip

class CLIPWrapper(nn.Module):
    def __init__(self, model: str, 
                 pretrained: str, 
                 clip_dim: int, 
                 latent_dim: int,
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
        self.mlp1 = nn.Linear(clip_dim, latent_dim)
        self.mlp2 = nn.Linear(clip_dim, latent_dim)
        self.fuser = nn.Linear(latent_dim*2, latent_dim)
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
    
    # def encode_query(self, queries: torch.Tensor):
    #     '''
    #         questions: The questions to encode
    #         return: question embedding(s)
    #     '''
    #     return self.mlp3(self.model.encode_text(queries))
    
    
    def forward(self, imgs, texts):
        return self.fuser(torch.cat((self.encode_image(imgs), self.encode_text(texts)), dim=-1))
    

