import sys
sys.path.append('..')
import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import clip
# from Text_Prompt import *
# from tools import *
from einops import rearrange, repeat
# from ctrgcn import *
# from model.transformer_encoder import Model as transformer_encoder



class TextPrompt(nn.Module):
    def __init__(self, head=['ViT-B/32']) :
        super(TextPrompt, self).__init__()
        self.head = head
        if 'ViT-B/32' in self.head:
            # self.attention_layer= nn.TransformerEncoderLayer(d_model=512, dim_feedforward=128, nhead=1, batch_first=True)
            # self.attention = nn.TransformerEncoder(self.attention_layer, num_layers=1)
            # self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
            self.fc = nn.Linear(512, 60)
        if 'ViT-B/16' in self.head:
            # self.attention_layer= nn.TransformerEncoderLayer(d_model=512, dim_feedforward=128, nhead=1, batch_first=True)
            # self.attention = nn.TransformerEncoder(self.attention_layer, num_layers=1)
            # self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
            self.fc = nn.Linear(512, 60)
        if 'ViT-L/14' in self.head:
            # self.attention_layer= nn.TransformerEncoderLayer(d_model=768, dim_feedforward=128, nhead=1, batch_first=True)
            # self.attention = nn.TransformerEncoder(self.attention_layer, num_layers=1)
            # self.cls_token = nn.Parameter(torch.randn(1, 1, 768))
            self.fc = nn.Linear(768, 60)
        if 'ViT-L/14@336px' in self.head:
            # self.attention_layer= nn.TransformerEncoderLayer(d_model=768, dim_feedforward=128, nhead=1, batch_first=True)
            # self.attention = nn.TransformerEncoder(self.attention_layer, num_layers=1)
            # self.cls_token = nn.Parameter(torch.randn(1, 1, 768))
            self.fc = nn.Linear(768, 60)
        if 'RN50x64' in self.head:
            # self.attention_layer= nn.TransformerEncoderLayer(d_model=1024, dim_feedforward=128, nhead=1, batch_first=True)
            # self.attention = nn.TransformerEncoder(self.attention_layer, num_layers=1)
            # self.cls_token = nn.Parameter(torch.randn(1, 1, 1024))
            self.fc = nn.Linear(1024, 60)
        if 'RN50x16' in self.head:
            # self.attention_layer= nn.TransformerEncoderLayer(d_model=768, dim_feedforward=128, nhead=1, batch_first=True)
            # self.attention = nn.TransformerEncoder(self.attention_layer, num_layers=1)
            # self.cls_token = nn.Parameter(torch.randn(1, 1, 768))
            self.fc = nn.Linear(768, 60)
        # self.logit_scale_text = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x):
        n, t, c = x.size()
        baseline = x.mean(1)
        # cls_tokens = self.cls_token.expand(n, -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        # x = self.attention(x)
        # x = x[:,0,:]
        # x = x.mean(1)
        # logits = self.fc(x)
        # x = x.mean(1)
        # logits = self.fc(x)
        return baseline
    
class ImagePrompt(nn.Module):
    def __init__(self, head=['ViT-B/32']) :
        super(ImagePrompt, self).__init__()
        self.head = head
        if 'ViT-B/32' in self.head:
            self.attention_layer= nn.TransformerEncoderLayer(d_model=512, dim_feedforward=128, nhead=1, batch_first=True)
            self.attention = nn.TransformerEncoder(self.attention_layer, num_layers=1)
            self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
            self.fc = nn.Linear(512, 120)
        if 'ViT-B/16' in self.head:
            self.attention_layer= nn.TransformerEncoderLayer(d_model=512, dim_feedforward=128, nhead=1, batch_first=True)
            self.attention = nn.TransformerEncoder(self.attention_layer, num_layers=1)
            self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
            self.fc = nn.Linear(512, 120)
        if 'ViT-L/14' in self.head:
            self.attention_layer= nn.TransformerEncoderLayer(d_model=768, dim_feedforward=128, nhead=1, batch_first=True)
            self.attention = nn.TransformerEncoder(self.attention_layer, num_layers=1)
            self.cls_token = nn.Parameter(torch.randn(1, 1, 768))
            self.fc = nn.Linear(768, 120)
        if 'ViT-L/14@336px' in self.head:
            self.attention_layer= nn.TransformerEncoderLayer(d_model=768, dim_feedforward=128, nhead=1, batch_first=True)
            self.attention = nn.TransformerEncoder(self.attention_layer, num_layers=1)
            self.cls_token = nn.Parameter(torch.randn(1, 1, 768))
            self.fc = nn.Linear(768, 60)
        if 'RN50x64' in self.head:
            self.attention_layer= nn.TransformerEncoderLayer(d_model=1024, dim_feedforward=128, nhead=1, batch_first=True)
            self.attention = nn.TransformerEncoder(self.attention_layer, num_layers=1)
            self.cls_token = nn.Parameter(torch.randn(1, 1, 1024))
            self.fc = nn.Linear(1024, 120)
        if 'RN50x16' in self.head:
            self.attention_layer= nn.TransformerEncoderLayer(d_model=768, dim_feedforward=128, nhead=1, batch_first=True)
            self.attention = nn.TransformerEncoder(self.attention_layer, num_layers=1)
            self.cls_token = nn.Parameter(torch.randn(1, 1, 768))
            self.fc = nn.Linear(768, 120)
        # self.attention_layer= nn.TransformerEncoderLayer(d_model=512, dim_feedforward=128, nhead=1, batch_first=True)
        # self.attention = nn.TransformerEncoder(self.attention_layer, num_layers=1)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        # self.fc = nn.Linear(512, 60)
        # self.logit_scale_text = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x):
        n, t, c = x.size()
        baseline = x.mean(1)
        cls_tokens = self.cls_token.expand(n, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.attention(x)
        x = x[:,0,:]
        # x = x.mean(1)
        logits = self.fc(x)
        # x = x.mean(1)
        # logits = self.fc(x)
        return baseline, x, logits

class TextCLIP(nn.Module):
    def __init__(self, head=['ViT-B/32']) :
        super(TextCLIP, self).__init__()
        self.text_prompt = TextPrompt(head)
        for p in self.parameters():
            p.requires_grad = False

        self.head = head
        if 'ViT-B/32' in self.head:
            self.fc = nn.Linear(512, 512)
        if 'ViT-B/16' in self.head:
            self.fc = nn.Linear(512, 512)
        if 'ViT-L/14' in self.head:
            self.fc = nn.Linear(768, 768)
        if 'ViT-L/14@336px' in self.head:
            self.fc = nn.Linear(768, 768)
        if 'RN50x64' in self.head:
            self.fc = nn.Linear(1024, 1024)
        if 'RN50x16' in self.head:
            self.fc = nn.Linear(768, 768)
        self.logit_scale_text = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # self.bn = nn.BatchNorm1d(512)

    def forward(self, x):
        n, t, c = x.size()
        # x = x.permute(0,2,1)
        # x = self.bn(x)
        # x = x.permute(0,2,1)
        _, x, logits = self.text_prompt(x)
        x = self.fc(x)
        return x, self.logit_scale_text, logits

class ImageCLIP(nn.Module):
    def __init__(self, head=['ViT-B/32']) :
        super(ImageCLIP, self).__init__()
        self.rgb_prompt = ImagePrompt(head)
        for p in self.parameters():
            p.requires_grad = False
        self.head = head
        if 'ViT-B/32' in self.head:
            self.fc = nn.Linear(512, 512)
        if 'ViT-B/16' in self.head:
            self.fc = nn.Linear(512, 512)
        if 'ViT-L/14' in self.head:
            self.fc = nn.Linear(768, 768)
        if 'ViT-L/14@336px' in self.head:
            self.fc = nn.Linear(768, 768)
        if 'RN50x64' in self.head:
            self.fc = nn.Linear(1024, 1024)
        if 'RN50x16' in self.head:
            self.fc = nn.Linear(768, 768)
        # self.fc = nn.Linear(512, 512)
        self.logit_scale_image = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x):
        n, t, c = x.size()
        _, x, logits = self.rgb_prompt(x)
        x = self.fc(x)
        return x, self.logit_scale_image, logits
    

if __name__ == "__main__":
    model = TextPrompt()
    x = torch.rand(10, 16, 512)
    y = model(x)
    print(y.size())