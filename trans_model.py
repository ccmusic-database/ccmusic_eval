import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torchvision.models as models
from modelscope.msdatasets import MsDataset
from utils import download
import itertools
TRAIN_MODES = ["linear_probe", "full_finetune", "no_pretrain"]

class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='bilinear', align_corners=False):
        super(Interpolate, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor,
                             mode=self.mode, align_corners=self.align_corners)

class t_Net:
    def __init__(
        self,
        backbone: str,
        train_mode: int,
        cls_num: int,
        ori_T: int,
        imgnet_ver="v1",
        weight_path="",
    ):
        if not train_mode in range(len(TRAIN_MODES)):
            raise ValueError(f"Unsupported training mode {train_mode}.")

        if not hasattr(models, backbone):
            raise ValueError(f"Unsupported model {backbone}.")

        self.imgnet_ver = imgnet_ver
        self.training = bool(weight_path == "")
        self.full_finetune = bool(train_mode > 0)
        self.type, self.weight_url, self.input_size = self._model_info(backbone)
        self.model: torch.nn.Module = eval("models.%s()" % backbone)
        
        self.ori_T = ori_T
        
        if self.type == 'vit':
            self.hidden_dim = self.model.hidden_dim
            self.class_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        
        elif self.type == 'swin_transformer':
            self.hidden_dim = 768

        self.cls_num = cls_num
        if self.training:
            if train_mode < 2:
                weight_path = self._download_model(self.weight_url)
                checkpoint = (
                    torch.load(weight_path)
                    if torch.cuda.is_available()
                    else torch.load(weight_path, map_location="cpu")
                )
                self.model.load_state_dict(checkpoint, False)

            for parma in self.model.parameters():
                parma.requires_grad = self.full_finetune

            self._set_classifier()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                self.classifier = self.classifier.cuda()
            self.model.train()

        else:
            self._set_classifier()
            checkpoint = (
                torch.load(weight_path)
                if torch.cuda.is_available()
                else torch.load(weight_path, map_location="cpu")
            )
            self.model.load_state_dict(checkpoint['model'], False)
            self.classifier.load_state_dict(checkpoint['classifier'], False)
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                self.classifier = self.classifier.cuda()
            self.model.eval()
    


    def _get_backbone(self, backbone_ver, backbone_list):
        for backbone_info in backbone_list:
            if backbone_ver == backbone_info["ver"]:
                return backbone_info

        raise ValueError("[Backbone not found] Please check if --model is correct!")

    def _model_info(self, backbone: str):
        backbone_list = MsDataset.load(
            "monetjoe/cv_backbones",
            split=self.imgnet_ver,
            cache_dir="./__pycache__",
            # download_mode="force_redownload",
        )
        backbone_info = self._get_backbone(backbone, backbone_list)
        return (
            str(backbone_info["type"]),
            str(backbone_info["url"]),
            int(backbone_info["input_size"]),
            # int(backbone_info['hidden_dim'])
        )

    def _download_model(self, weight_url: str, model_dir="./__pycache__"):
        weight_path = f'{model_dir}/{weight_url.split("/")[-1]}'
        os.makedirs(model_dir, exist_ok=True)
        if not os.path.exists(weight_path):
            download(weight_url, weight_path)

        return weight_path

    def _create_classifier(self):
        original_T_size = self.ori_T
        self.avgpool = nn.AdaptiveAvgPool2d((1, None)) # F -> 1 
        upsample_module = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1, None)), # F -> 1
            
            nn.ConvTranspose2d(self.hidden_dim, 256, kernel_size=(1,4), stride=(1,2), padding=(0,1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            nn.ConvTranspose2d(256, 128, kernel_size=(1,4), stride=(1,2), padding=(0,1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 64, kernel_size=(1,4), stride=(1,2), padding=(0,1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, kernel_size=(1,4), stride=(1,2), padding=(0,1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            # input for Interp: [bsz, C, 1, T]
            Interpolate(size=(1, original_T_size), mode='bilinear', align_corners=False),
            # classifier
            nn.Conv2d(32, 32, kernel_size=(1,1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, self.cls_num, kernel_size=(1,1)) 
        )

        return upsample_module

    def _set_classifier(self):
        #### set custom classifier ####
        if self.type == "vit":
            self.classifier = self._create_classifier()

        elif self.type == 'swin_transformer':
            self.classifier = self._create_classifier()


        for parma in self.classifier.parameters():
            parma.requires_grad = True

    def get_input_size(self):
        return self.input_size

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.cuda()

        if self.type == 'vit':
            x = self.model._process_input(x)
            batch_class_token = self.class_token.expand(x.size(0), -1, -1).cuda()
            x = torch.cat([batch_class_token, x], dim=1)
            x = self.model.encoder(x)
            x = x[:, 1:].permute(0, 2, 1) 
            x = x.unsqueeze(2)  
            x = self.classifier(x).squeeze() # # x shape: [bsz, hidden_dim, 1, seq_len]
            return x

        elif self.type == 'swin_transformer':
            x = self.model.features(x) # [B, H, W, C]
            x = x.permute(0, 3, 1, 2)
            x = self.avgpool(x) # [B, C, 1, W]
            x = self.classifier(x).squeeze()
            return x



    def parameters(self):
        if self.full_finetune:
            return itertools.chain(self.model.parameters(), self.classifier.parameters())
        else:
            return self.classifier.parameters()

    def state_dict(self):
        return {
        'model': self.model.state_dict(),
        'classifier': self.classifier.state_dict()
    }
