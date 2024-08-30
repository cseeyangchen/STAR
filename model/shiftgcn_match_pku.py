import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.tgcn import ConvTemporalGraphical
from model.utils.graph import Graph
import math
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from sklearn import manifold
import numpy as np
from scipy.special import binom



import sys
sys.path.append("./model/Temporal_shift/")

from cuda.shift import Shift




def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def conv_init(conv):
    nn.init.kaiming_normal(conv.weight, mode='fan_out')
    nn.init.constant(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant(bn.weight, scale)
    nn.init.constant(bn.bias, 0)


class tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class Shift_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(Shift_tcn, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        bn_init(self.bn2, 1)
        self.relu = nn.ReLU(inplace=True)
        self.shift_in = Shift(channel=in_channels, stride=1, init_scale=1)
        self.shift_out = Shift(channel=out_channels, stride=stride, init_scale=1)

        self.temporal_linear = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.kaiming_normal(self.temporal_linear.weight, mode='fan_out')

    def forward(self, x):
        x = self.bn(x)
        # shift1
        x = self.shift_in(x)
        x = self.temporal_linear(x)
        x = self.relu(x)
        # shift2
        x = self.shift_out(x)
        x = self.bn2(x)
        return x


class Shift_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(Shift_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.Linear_weight = nn.Parameter(torch.zeros(in_channels, out_channels, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0,math.sqrt(1.0/out_channels))

        self.Linear_bias = nn.Parameter(torch.zeros(1,1,out_channels,requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant(self.Linear_bias, 0)

        self.Feature_Mask = nn.Parameter(torch.ones(1,25,in_channels, requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant(self.Feature_Mask, 0)

        self.bn = nn.BatchNorm1d(25*out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

        index_array = np.empty(25*in_channels).astype(int)
        # index_array = np.empty(25*in_channels, dtype=int)
        for i in range(25):
            for j in range(in_channels):
                index_array[i*in_channels + j] = (i*in_channels + j + j*in_channels)%(in_channels*25)
        self.shift_in = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)

        index_array = np.empty(25*out_channels).astype(int)
        for i in range(25):
            for j in range(out_channels):
                index_array[i*out_channels + j] = (i*out_channels + j - j*out_channels)%(out_channels*25)
        self.shift_out = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)
        

    def forward(self, x0):
        n, c, t, v = x0.size()
        x = x0.permute(0,2,3,1).contiguous()

        # shift1
        x = x.view(n*t,v*c)
        x = torch.index_select(x, 1, self.shift_in)
        x = x.view(n*t,v,c)
        x = x * (torch.tanh(self.Feature_Mask)+1)

        x = torch.einsum('nwc,cd->nwd', (x, self.Linear_weight)).contiguous() # nt,v,c
        x = x + self.Linear_bias

        # shift2
        x = x.view(n*t,-1) 
        x = torch.index_select(x, 1, self.shift_out)
        x = self.bn(x)
        x = x.view(n,t,v,self.out_channels).permute(0,3,1,2) # n,c,t,v

        x = x + self.down(x0)
        x = self.relu(x)
        return x


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = Shift_gcn(in_channels, out_channels, A)
        self.tcn1 = Shift_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)



class SHIFTGCNModel(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(SHIFTGCNModel, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)

        x_match_feature = x.view(N, c_new, T//4, V, M)
        x_match_feature = x_match_feature.mean(4)
        
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        # x = self.fc(x)
        # return x, self.fc(x)
        return x_match_feature, x




class STGCNModel(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels,  num_class, graph_args,
                 edge_importance_weighting, head=['ViT-B/32'], num_point=25, num_person=2, graph=None, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

        # contrastive learning
        self.linear_head_text = nn.ModuleDict()

        self.logit_scale_text = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.head = head
        if 'ViT-B/32' in self.head:
            self.linear_head_text['ViT-B/32'] = nn.Linear(256,512)
            conv_init(self.linear_head_text['ViT-B/32'])
   
        if 'ViT-B/16' in self.head:
            self.linear_head_text['ViT-B/16'] = nn.Linear(256,512)
            conv_init(self.linear_head_text['ViT-B/16'])
            
        if 'ViT-L/14' in self.head:
            self.linear_head_text['ViT-L/14'] = nn.Linear(256,768)
            conv_init(self.linear_head_text['ViT-L/14'])
            # self.linear_head_image['ViT-L/14'] = nn.Linear(256,768)
            # conv_init(self.linear_head_image['ViT-L/14'])
        if 'ViT-L/14@336px' in self.head:
            self.linear_head_text['ViT-L/14@336px'] = nn.Linear(256,768)
            conv_init(self.linear_head_text['ViT-L/14@336px'])
            # self.linear_head_image['ViT-L/14@336px'] = nn.Linear(256,768)
            # conv_init(self.linear_head_image['ViT-L/14@336px'])
        if 'RN50x64' in self.head:
            self.linear_head_text['RN50x64'] = nn.Linear(256,1024)
            conv_init(self.linear_head_text['RN50x64'])
            # self.linear_head_image['RN50x64'] = nn.Linear(256,1024)
            # conv_init(self.linear_head_image['RN50x64'])
        if 'RN50x16' in self.head:
            self.linear_head_text['RN50x16'] = nn.Linear(256,768)
            conv_init(self.linear_head_text['RN50x16'])
            # self.linear_head_image['RN50x16'] = nn.Linear(256,768)
            # conv_init(self.linear_head_image['RN50x16'])

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)


        # N*M,C,T,V
        c_new = x.size(1)

       

        x_match_feature = x.view(N, c_new, T//4, V, M)
        x_match_feature = x_match_feature.mean(4)

        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        text_feature_dict = dict()
        # image_feature_dict = dict()

        for name in self.head:
            text_feature_dict[name] = self.linear_head_text[name](x)   # global text feature
          

        return x_match_feature, x, text_feature_dict, self.logit_scale_text
    

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature




class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model.float()
        self.fc = nn.Linear(512, 512)

    def forward(self, x):
        x = self.model.encode_text(x)
        x = self.fc(x)
        return x
    


class ImageCLIP(nn.Module):
    def __init__(self) :
        super(ImageCLIP, self).__init__()
        self.fc = nn.Linear(512, 512)

    def forward(self, x):
        n, t, c = x.size()
        x = self.fc(x.mean(1))
        return x



class ModelMatch(nn.Module):
    def __init__(self,num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, head=['ViT-B/32'], k=0, body_part=6):
        super(ModelMatch, self).__init__()
        # pretrain model
        self.pretraining_model = SHIFTGCNModel(num_class=num_class,num_point=num_point, num_person=num_person, graph=graph, graph_args=graph_args, in_channels=in_channels)
        for p in self.parameters():
            p.requires_grad = False
        # match network
        # body part index
        self.body_part = body_part
        self.body_part_index_list = nn.ParameterList()
        # head hand arm hip leg foot 
        self.body_part_index_list.append(nn.Parameter(torch.Tensor([2,3,20]).long(), requires_grad=False))
        self.body_part_index_list.append(nn.Parameter(torch.Tensor([7,11,21,22,23,24]).long(), requires_grad=False))
        self.body_part_index_list.append(nn.Parameter(torch.Tensor([4,5,6,8,9,10]).long(), requires_grad=False))
        self.body_part_index_list.append(nn.Parameter(torch.Tensor([0,1]).long(), requires_grad=False))
        self.body_part_index_list.append(nn.Parameter(torch.Tensor([12,13,16,17]).long(), requires_grad=False))
        self.body_part_index_list.append(nn.Parameter(torch.Tensor([14,15,18,19]).long(), requires_grad=False))
        # body part spatial temporal attention network
        self.body_part_st_attention_networks = nn.ModuleList()
        # self.body_part_st_attention_networks1 = nn.ModuleList()
        self.body_part_mapping_matrix1 = nn.ModuleList()
        self.body_part_mapping_matrix2 = nn.ModuleList()
        self.body_part_prompts = nn.ParameterList()
        # self.body_part_prompts1 = nn.ParameterList()
        self.body_part_w = nn.ParameterList()
        self.pd_mapping_matrix1 = nn.ModuleList()
        self.pd_mapping_matrix2 = nn.ModuleList()
        self.part_weights_factor = nn.Parameter(torch.ones(1, 6), requires_grad=False)
        self.gcn_part_classifier = nn.ModuleList()
        self.pd_prompt = nn.ParameterList()
        for _ in range(self.body_part):
            self.body_part_st_attention_networks.append(nn.MultiheadAttention(embed_dim=768, kdim=256, vdim=256, num_heads=4, batch_first=True))  # head:4
            self.body_part_mapping_matrix1.append(nn.Linear(768, 768))
            self.body_part_mapping_matrix2.append(nn.Linear(768, 768))
            self.body_part_prompts.append(nn.Parameter(nn.init.normal_(torch.empty(100, 768)), requires_grad=True))   # 100 768
            self.body_part_w.append(nn.Parameter(nn.init.normal_(torch.empty(768, 768)), requires_grad=True))
            self.pd_mapping_matrix1.append(nn.Linear(768,256))
            self.pd_mapping_matrix2.append(nn.Linear(256,100))
            self.gcn_part_classifier.append(nn.Linear(256,100))
            self.pd_prompt.append(nn.Parameter(nn.init.normal_(torch.empty(51, 768)), requires_grad=True))
        self.relu = nn.ReLU()
        self.loss_mse = torch.nn.MSELoss()
        self.sim_loss_mse = torch.nn.MSELoss(reduce=False)
        # conditional vae
        self.cvae = nn.ModuleList()
        self.cvae_memory = nn.ParameterList()
        # classfication
        self.part_classification = nn.ModuleList()
        self.class_loss = nn.CrossEntropyLoss()
        
        self.crossmodal_align1 = nn.Linear(768, 256)
       
        self.crossmodal_align2 = nn.Linear(256, 100)

        self.kl = nn.KLDivLoss(reduction='batchmean')
        # part-global
        self.global_fc1 = nn.Linear(100*6, 256)
        self.global_fc2 = nn.Linear(256, 100)
        # gcn 
        self.gcn_flobal_classifier = nn.Linear(256, 51)
        
            
    def forward(self, x, st_attributes, part_des_feature, label_language, train_flag, part_language_seen):
        gcn_x, _ = self.pretraining_model(x)
        n,c,t,v = gcn_x.size()
        # spatial temporal attention
        part_visual_feature = []
        global_visual_feature = []
        part_reconstruction_feature = []
        part_mu_feature = []
        part_logvar_feature = []
        sim_score = []
        memory_weights = []
        class_prob = []
        part_visual_feature_pd = []
        partz_feature = []
        part_des_mapping_feature = []
        gcn_feature = []
        ske_feature = []
        gcn_global = gcn_x.mean(3).mean(2)
        for i, part_name in enumerate(["head", "hand", "arm", "hip", "leg", "foot"]):
            # normalize
            part_feature_original = gcn_x[:,:,:,self.body_part_index_list[i]].view(n,c,-1).permute(0,2,1)
            gcn_feature.append(self.gcn_part_classifier[i](part_feature_original.mean(1)).unsqueeze(1))

            part_feature = part_feature_original

            st_attributes_feature, _ = self.body_part_st_attention_networks[i](self.body_part_prompts[i].unsqueeze(0).expand(n, -1, -1), part_feature, part_feature)
            embedding_augamented = self.relu(self.body_part_mapping_matrix1[i](st_attributes_feature))
            embedding_augamented = self.body_part_mapping_matrix2[i](embedding_augamented)

           
            embedding_semantic = torch.einsum('iv, vf, bif->bi',self.body_part_prompts[i], self.body_part_w[i], embedding_augamented)
            # append feature
            global_visual_feature.append(embedding_semantic.unsqueeze(1))
            ske_feature.append(embedding_semantic)
            part_visual_feature.append(embedding_augamented.unsqueeze(1))
            part_visual_feature_pd.append(embedding_augamented.unsqueeze(1))
            
            part_des_mapping = self.relu(self.pd_mapping_matrix1[i](part_des_feature[:,i,:]))
            part_des_mapping = self.pd_mapping_matrix2[i](part_des_mapping)
            part_des_mapping_feature.append(part_des_mapping.unsqueeze(1))
        part_visual_feature = torch.cat(part_visual_feature, dim=1)
        gcn_feature = torch.cat(gcn_feature, dim=1)
        part_visual_feature_pd = torch.cat(part_visual_feature_pd, dim=1)
        global_visual_feature = torch.cat(global_visual_feature, dim=1)
        
        label_language = self.crossmodal_align2(label_language)
        part_des_mapping_feature = torch.cat(part_des_mapping_feature, dim=1)

        ske_feature = torch.cat(ske_feature, dim=1)
        ske_feature = self.relu(self.global_fc1(ske_feature))
        ske_feature = self.global_fc2(ske_feature)
        # gcn global
        gcn_global = self.gcn_flobal_classifier(gcn_global)
        global_semantic = torch.einsum('bpd,qp->bdq',global_visual_feature,self.part_weights_factor).squeeze(2)
        return part_visual_feature, part_visual_feature_pd, global_visual_feature, part_reconstruction_feature, part_mu_feature, part_logvar_feature, sim_score,memory_weights, class_prob, label_language, part_des_mapping_feature, gcn_feature, gcn_global, ske_feature, global_semantic
    
    
        

    def loss_cal(self, part_visual, global_visual, part_language, part_language_seen,part_language_seen_unseen,
                 label_langauge, mse_label_language, true_seen_label, all_label_language, unseen_label,
                 part_reconstruction_embedding,part_mu_feature, part_logvar_feature, sim_score,memory_weights, class_prob, part_visual_feature_pd,
                 all_true_label_array, part_des_mapping_feature, gcn_feature, gcn_global, ske_feature,
                 global_semantic):
        n, _, _, _ = part_visual.size()
        
        label_langauge = self.relu(self.crossmodal_align1(label_langauge))
        label_langauge = self.crossmodal_align2(label_langauge)
        label_langauge = label_langauge.expand(n, -1, -1)  # n 55 768
        all_label_language = self.relu(self.crossmodal_align1(all_label_language))
        all_label_language = self.crossmodal_align2(all_label_language)
        global_vl_pred_score = torch.einsum('nk,njk->nj', F.normalize(global_semantic, dim=1, p=2), F.normalize(all_label_language.expand(n, -1, -1), dim=2, p=2))  # n 6 60
        seen_label = list(set(range(51))-set(unseen_label))
        loss_calibration_unseen = torch.tensor(1)
        global_vl_pred_score = F.softmax(global_vl_pred_score[:,seen_label], dim=1)
        global_vl_pred_score = torch.log(global_vl_pred_score+1e-12)
        loss_global_ce = -torch.einsum('nc,nc->n',global_vl_pred_score, true_seen_label.float())
        loss_global_ce = sum(loss_global_ce)/n
        loss_kl_divergence = torch.tensor(1)


        loss_gcn_classify = sum(-torch.einsum('bc,bc->b',torch.log(F.softmax(gcn_global, dim=1)[:,seen_label]+1e-12), true_seen_label.float()))/n


        loss_part_ce = []
        for i in range(6):
            tmp = part_language_seen_unseen[:,i,:] + self.pd_prompt[i]
            tmp = self.relu(self.pd_mapping_matrix1[i](tmp))
            tmp = self.pd_mapping_matrix2[i](tmp)
            global_pd_pred_score = torch.einsum('ni,ji->nj', F.normalize(global_visual[:,i,:], dim=1, p=2), F.normalize(tmp, dim=1, p=2))  # n 60

            global_pd_pred_score = F.softmax(global_pd_pred_score[:,seen_label], dim=1)

            global_pd_pred_score = torch.log(global_pd_pred_score+1e-12)
            loss_part = torch.mean(-torch.einsum('nc,nc->n',global_pd_pred_score, true_seen_label.float()),dim=0)
            loss_part_ce.append(loss_part)
        loss_part_ce = sum(loss_part_ce)/6


        loss_pd_label = []
        for i in range(6):

            tmp = part_language_seen_unseen[:,i,:] + self.pd_prompt[i]
            tmp = self.relu(self.pd_mapping_matrix1[i](tmp))
            tmp = self.pd_mapping_matrix2[i](tmp)  # 60 50
            sim = F.softmax(torch.einsum('cd,kd->ck',tmp, all_label_language), dim=1)  # 60 60
            sim = -torch.log(torch.diag(sim)+1e-12)
            loss_pd_label.append(torch.mean(sim, dim=0))
        loss_pd_label = sum(loss_pd_label)/6
        
        ske_pre_score = torch.einsum('nd,ncd->nc', ske_feature, all_label_language.unsqueeze(0).expand(n, -1, -1))  # n 55
        ske_pre_score = F.softmax(ske_pre_score, dim=1)
        ske_pre_score = torch.log(ske_pre_score+1e-12)
        loss_ske_ce = torch.sum(-torch.einsum('nc,nc->n',ske_pre_score, all_true_label_array.float()),dim=0)/n
        # print("loss_ske_ce:",loss_ske_ce)


        loss_align_lpd = torch.tensor(1)
        loss_align_spd = torch.tensor(1)
       
        loss_mse_align = self.loss_mse(F.normalize(global_visual, dim=2, p=2), F.normalize(part_des_mapping_feature,dim=2,p=2))
        
        loss_mse_lb_pd = []
        for i in range(6):

            tmp = part_language_seen_unseen[:,i,:] + self.pd_prompt[i]
            tmp = self.relu(self.pd_mapping_matrix1[i](tmp))
            tmp = self.pd_mapping_matrix2[i](tmp)  # 60 50

            loss = self.loss_mse(F.normalize(tmp,dim=1, p=2), F.normalize(all_label_language,dim=1, p=2))
            loss_mse_lb_pd.append(loss)
        loss_mse_lb_pd = sum(loss_mse_lb_pd)/6

        loss_mse_ske_lb = torch.tensor(1)

        loss_factor = torch.tensor(1)

        loss_prompt = torch.tensor(1)

        loss_cvae_reconstruction = torch.tensor(1)
        loss_cvae_kld = torch.tensor(1)
    
        loss_classification = torch.tensor(1)
        loss_classification_unseen = torch.tensor(1)

        loss_global_ce_semantic = torch.tensor(1)

        return loss_global_ce, loss_cvae_reconstruction, loss_cvae_kld, loss_classification, loss_mse_align, loss_align_lpd, loss_align_spd, loss_prompt, loss_global_ce_semantic, loss_kl_divergence, loss_calibration_unseen, loss_part_ce, loss_pd_label, loss_classification_unseen, loss_ske_ce, loss_mse_ske_lb, loss_mse_lb_pd, loss_factor, loss_gcn_classify
    
    def get_zsl_acc(self, global_visual, part_visual, label_langauge, part_language_unseen, true_label_list, unseen_label, part_des_mapping_feature,
                    ske_feature, global_semantic,gcn_global,gcn_feature):
        n,_,_ = global_visual.size()
        part_language_unseen = part_language_unseen.permute(1,0,2)
        part_language_unseen = part_language_unseen.unsqueeze(0).expand(n,-1,-1,-1)
    
        label_langauge = self.relu(self.crossmodal_align1(label_langauge))
        
        label_langauge = self.crossmodal_align2(label_langauge)
        
        label_langauge = label_langauge.expand(n, -1, -1)  #
        
        global_vl_pred = torch.einsum('nk,njk->nj', F.normalize(global_semantic, dim=1, p=2), F.normalize(label_langauge, dim=2, p=2))  # n 6 60
        
        global_vl_pred = F.softmax(global_vl_pred, dim=1)


        global_vl_pred_idx = torch.max(global_vl_pred, dim=1)[1].data.cpu().numpy()
        true_label_list = torch.max(true_label_list, dim=1)[1].data.cpu().numpy()
        return global_vl_pred_idx, true_label_list
    

    def get_gzsl_acc(self, global_visual, part_visual, label_langauge, part_language_unseen_seen,true_label_list, unseen_label, 
                     sim_score, part_mu_feature, part_logvar_feature, part_des_mapping_feature, ske_feature, global_semantic,gcn_global,gcn_feature):
        n,_,_ = global_visual.size()
        
        part_language_unseen_seen = part_language_unseen_seen.permute(1,0,2)
        part_language_unseen_seen = part_language_unseen_seen.unsqueeze(0).expand(n,-1,-1,-1)
        
        label_langauge = self.relu(self.crossmodal_align1(label_langauge))
       
        label_langauge = self.crossmodal_align2(label_langauge)

       

        label_langauge = label_langauge.expand(n, -1, -1)
        
        global_vl_pred = torch.einsum('nk,njk->nj', F.normalize(global_semantic, dim=1, p=2), F.normalize(label_langauge, dim=2,p=2))  # n 6 60
       
        seen_label = list(set(range(51))-set(unseen_label))

       

        global_vl_pred_idx = torch.max(global_vl_pred, dim=1)[1].data.cpu().numpy()
        
        true_label_list = torch.max(true_label_list, dim=1)[1].data.cpu().numpy()
        threshold_seen_list = []
        threshold_unseen_list = []

        
        return (global_vl_pred_idx, true_label_list, threshold_seen_list, threshold_unseen_list, global_vl_pred)
    
   
        





  
        
       
        





        