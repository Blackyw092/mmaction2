from mmengine.model import BaseModule, Sequential
from .custom_MAE2 import VisionTransformer,load_MAE2
from .custom_clip import load,CLIP
from .custom_Visual_Prompt import visual_prompt,clsHead
from mmaction.registry import MODELS
from functools import partial
import torch.nn as nn
import torch
import time
from mmengine.logging import MessageHub


class ImageCLIP(nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self,image):
        return self.model.encode_image(image)

@MODELS.register_module()
class clip_MAE2(BaseModule):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,


                 ):
        super().__init__()
        self.MAE2 =  VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        self.MAE2 = load_MAE2(self.MAE2)

        clip, clip_state_dict = load('ViT-B/16', device='cuda', jit=False, T=8, dropout=0.0, emb_dropout=0.0, pretrain=True)
        self.clip_img = ImageCLIP(clip).float()
        self.fusion_model = visual_prompt('Transf', clip_state_dict, int(16/ 4))

        self.A = torch.nn.Parameter(torch.randn(1), requires_grad=True)

        self.A.data.fill_(0.25)


    def init_weights(self, pretrained=None):
        pass
    def forward(self, x):
        # start_time = time.time()
        images_v = x[:, :int(16 / 2), :]
        images_sk = x[:, int(16 / 2):, :]
        b, t, c, h, w = images_v.size()
        image_embedding_v = self.clip_img(images_v.reshape(-1, c, h, w))
        image_embedding_sk = self.MAE2(images_sk)

        image_embedding_v = image_embedding_v.reshape(b, t, -1)
        image_embedding_v = self.fusion_model(image_embedding_v)
        image_embedding =  (1-self.A)* image_embedding_v +  self.A * image_embedding_sk
        # end_time = time.time()
        # duration = end_time - start_time
        # print(f"前向传播所耗时间: {duration} seconds")
        message_hub = MessageHub.get_current_instance()
        message_hub.update_scalar('比例', self.A)
        return image_embedding