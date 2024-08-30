import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
# from model.baseline import TextCLIP
# from model.text_classification import ImageCLIP
from model.text_classification import TextCLIP
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob
import clip
from PIL import Image
# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from torchlight import DictAction
from tools import *
from Text_Prompt import *
# from KLLoss import KLLoss
import matplotlib.pyplot as plt

import pandas as pd
from sklearn import manifold
import numpy as np
from scipy.special import binom



unseen_classes = [1, 9, 20, 34, 50]  # pkuv1_46/5_split
# unseen_classes = [3,7,11,15,19,21,25,31,33,36,43,48]  # pkuv1_39/12_split
seen_classes = list(set(range(51))-set(unseen_classes))  # ntu60
train_label_dict = {}
for idx, l in enumerate(seen_classes):
    tmp = [0] * len(seen_classes)
    tmp[idx] = 1
    train_label_dict[l] = tmp
test_zsl_label_dict = {}
for idx, l in enumerate(unseen_classes):
    tmp = [0] * len(unseen_classes)
    tmp[idx] = 1
    test_zsl_label_dict[l] = tmp
test_gzsl_label_dict = {}
for idx, l in enumerate(range(51)):
    tmp = [0] * 51
    tmp[idx] = 1
    test_gzsl_label_dict[l] = tmp


scaler = torch.cuda.amp.GradScaler()

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        # extract attributes feature and action descriptions features
        clip_output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.action_descriptions = torch.load('/DATA3/cy/STAR/data/text_feature/pkuv1_semantic_feature_dict_gpt35.tar')
        _ = torch.load('/DATA3/cy/STAR/data/text_feature/pkuv1_spatial_attribute_feature_dict_gpt35.tar')
        self.attribute_features_dict = torch.load('/DATA3/cy/STAR/data/text_feature/pkuv1_spatial_temporal_attribute_feature_dict_gpt35.tar')
        print('Extract CLIP Attributes Features and Action Descriptions Features Successful!')
        # load model
        self.load_model()

        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
            self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

        self.model = self.model.cuda(self.output_device)

        # load skeleton action recognition model
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)
        

        
    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args)
        self.loss_mse = nn.MSELoss().cuda(output_device)

        if self.arg.weights:
            self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict([["pretraining_model."+k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)
        print("Load model done.")


        
        
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            # print(list(self.model_text_dict))
            self.optimizer = optim.SGD(
                filter(lambda p:p.requires_grad, self.model.parameters()),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()
        
    
    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
            print("Load train data done.")
        self.data_loader['test_zsl'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_zsl_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)
        print("Load zsl test data done.")
        self.data_loader['test_gzsl'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_gzsl_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)
        print("Load gzsl test data done.")
    
    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()
    
    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time
    
    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)

        loss_value = []
        loss_part_infonce_value = []
        loss_global_ce_value = []
        loss_calibration_unseen_value = []
        loss_global_mse_value = []
        loss_part_mse_value = []
        loss_part_ce_value = []
        loss_cali_ce_value = []
        loss_cvae_reconstruction_value = []
        loss_sim_score_value = []
        loss_classification_value = []
        loss_cvae_kld_value = []
        loss_mse_align_value = []
        loss_align_lpd_value = []
        loss_align_spd_value = []
        loss_prompt_value = []
        loss_global_ce_semantic_value = []
        loss_kl_divergence_value = []
        loss_pd_label_value = []
        loss_classification_unseen_value = []
        loss_ske_ce_value = []
        loss_mse_ske_lb_value = []
        loss_mse_lb_pd_value = []
        loss_factor_value = []
        loss_gcn_classify_value = []

        loss_value_ce = []
        loss_value_img = []
        loss_value_text = []
        loss_value_img_mse = []
        loss_value_text_mse = []
        loss_value_mse = []
        loss_value_cross_modal_align = []
        
        acc_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)


        for batch_idx, (data, label, index) in enumerate(process):         
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                b,_,_,_,_ = data.size()
            timer['dataloader'] += self.split_time()
            self.optimizer.zero_grad()

            # forward
            # with torch.cuda.amp.autocast():
            part_language = []
            # label_language = []
            for i, part_name in enumerate(["head", "hand", "arm", "hip", "leg", "foot"]):
                part_language.append(self.action_descriptions[i+1].unsqueeze(1))
                # label_language.append(self.action_descriptions[i+1])
            part_language = torch.cat(part_language, dim=1).cuda(self.output_device)
            part_language_seen = part_language[seen_classes]
            part_language_seen_unseen = part_language[:51]
            # part_language = F.normalize(part_language, dim=2)
            prompt = torch.cat([self.model.pd_prompt[i].unsqueeze(1) for i in range(6)], dim=1)
            part_language = torch.cat([(part_language[l.item(),:,:].unsqueeze(0)+prompt[l.item(),:,:].unsqueeze(0)) for l in label], dim=0)
            # part_language = torch.cat([part_language[l.item(),:,:].unsqueeze(0) for l in label], dim=0)
            label_language_seen = self.action_descriptions[0].cuda(self.output_device)[seen_classes]
            # label_language = torch.cat([self.action_descriptions[0][l.item()].unsqueeze(0) for l in label], dim=0).cuda(self.output_device)
            label_language = torch.cat([self.action_descriptions[0][l.item()].unsqueeze(0) for l in label], dim=0).cuda(self.output_device)
            all_label_language = self.action_descriptions[0].cuda(self.output_device)[:51]
            true_label_array = torch.tensor([train_label_dict[l.item()] for l in label]).cuda(self.output_device)
            all_true_label_array = torch.tensor([test_gzsl_label_dict[l.item()] for l in label]).cuda(self.output_device)
            # self.attribute_features_dict = self.attribute_features_dict.cuda(self.output_device)

            

            part_visual_feature, part_visual_feature_pd, global_visual_feature, part_reconstruction_embedding, part_mu_feature, part_logvar_feature, sim_score, memory_weights, class_prob, label_language, part_des_mapping_feature, gcn_feature, gcn_global, ske_feature, global_semantic = self.model(data, self.attribute_features_dict, part_language, label_language,1, part_language_seen)
            loss_global_ce, loss_cvae_reconstruction, loss_cvae_kld,loss_classification, loss_mse_align, loss_align_lpd, loss_align_spd, loss_prompt, loss_global_ce_semantic, loss_kl_divergence, loss_calibration_unseen, loss_part_ce, loss_pd_label, loss_classification_unseen, loss_ske_ce, loss_mse_ske_lb, loss_mse_lb_pd, loss_factor, loss_gcn_classify = self.model.loss_cal(part_visual_feature, global_visual_feature, part_language, part_language_seen,part_language_seen_unseen, label_language_seen,
                                                                    label_language, true_label_array, all_label_language,unseen_classes,part_reconstruction_embedding,
                                                                    part_mu_feature, part_logvar_feature, sim_score, memory_weights, class_prob, part_visual_feature_pd, 
                                                                    all_true_label_array, part_des_mapping_feature, gcn_feature, gcn_global, ske_feature,
                                                                    global_semantic)
            
            loss = 0.1*loss_global_ce + loss_part_ce + 0.1*loss_pd_label  
            loss.backward()
            self.optimizer.step()

            loss_value.append(loss.data.item())
            # loss_part_infonce_value.append(loss_part_infonce.data.item())
            loss_global_ce_value.append(loss_global_ce.data.item())
            # loss_part_mse_value.append(loss_part_mse.data.item())
            loss_part_ce_value.append(loss_part_ce.data.item())
            # loss_cali_ce_value.append(loss_cali_ce.data.item())
            # loss_global_mse_value.append(loss_global_mse.data.item())
            loss_calibration_unseen_value.append(loss_calibration_unseen.data.item())
            loss_cvae_reconstruction_value.append(loss_cvae_reconstruction.data.item())
            # loss_sim_score_value.append(loss_sim_score.data.item())
            loss_classification_value.append(loss_classification.data.item())
            loss_cvae_kld_value.append(loss_cvae_kld.data.item())
            loss_mse_align_value.append(loss_mse_align.data.item())
            loss_align_lpd_value.append(loss_align_lpd.data.item())
            loss_align_spd_value.append(loss_align_spd.data.item())
            loss_prompt_value.append(loss_prompt.data.item())
            loss_global_ce_semantic_value.append(loss_global_ce_semantic.data.item())
            loss_kl_divergence_value.append(loss_kl_divergence.data.item())
            loss_pd_label_value.append(loss_pd_label.data.item())
            loss_classification_unseen_value.append(loss_classification_unseen.data.item())
            loss_ske_ce_value.append(loss_ske_ce.data.item())
            loss_mse_ske_lb_value.append(loss_mse_ske_lb.data.item())
            loss_mse_lb_pd_value.append(loss_mse_lb_pd.data.item())
            loss_factor_value.append(loss_factor.data.item())
            loss_gcn_classify_value.append(loss_gcn_classify.data.item())

            timer['model'] += self.split_time()

            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_global_ce', loss_global_ce.data.item(), self.global_step)
            # self.train_writer.add_scalar('loss_part_mse', loss_part_mse.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_part_ce', loss_part_ce.data.item(), self.global_step)
            # self.train_writer.add_scalar('loss_cali_ce', loss_cali_ce.data.item(), self.global_step)
            # self.train_writer.add_scalar('loss_global_mse', loss_global_mse.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_calibration_unseen', loss_calibration_unseen.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_cvae_reconstruction', loss_cvae_reconstruction.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_cvae_kld', loss_cvae_kld.data.item(), self.global_step)
            # self.train_writer.add_scalar('loss_sim_score', loss_sim_score.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_classification',loss_classification.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_mse_align',loss_mse_align.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_align_lpd',loss_align_lpd.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_align_spd',loss_align_spd.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_prompt',loss_prompt.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_global_ce_semantic',loss_global_ce_semantic.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_kl_divergence', loss_kl_divergence.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_pd_label', loss_pd_label.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_classification_unseen', loss_classification_unseen.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_ske_ce', loss_ske_ce.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_mse_ske_lb',loss_mse_ske_lb.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_mse_lb_pd',loss_mse_lb_pd.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_factor',loss_factor.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_gcn_classify', loss_gcn_classify.data.item(), self.global_step)
            # self.train_writer.add_scalar('loss_cross_modal_align', loss_cross_modal_align.data.item(), self.global_step)
            

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

            # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))



        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])

            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')


    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            threshold_unseen_value = []
            threshold_seen_value = []
            threshold_unseen_mu_value = []
            threshold_seen_mu_value = []
            threshold_unseen_logvar_value = []
            threshold_seen_logvar_value = []
            sim_score_list = []
            sim_matrix_list = []
            class_prob_list = []
            step = 0
            process = tqdm(self.data_loader[ln], ncols=40)



            for batch_idx, (data, label, index) in enumerate(process):
                with torch.no_grad():
                    # print(data.size())
                    b, _, _, _, _ = data.size()
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                
                    part_language = []
                    # label_language = []
                    for i, part_name in enumerate(["head", "hand", "arm", "hip", "leg", "foot"]):
                        part_language.append(self.action_descriptions[i+1].unsqueeze(1))
                        # label_language.append(self.action_descriptions[i+1])
                    part_language1 = torch.cat(part_language, dim=1).cuda(self.output_device)
                    # part_language = F.normalize(part_language, dim=2)
                    part_language = torch.cat([part_language1[l.item(),:,:].unsqueeze(0) for l in label], dim=0)
                    if ln == 'test_zsl':
                        label_language = self.action_descriptions[0].cuda(self.output_device)[unseen_classes]
                        sample_label_language = torch.cat([self.action_descriptions[0][l.item()].unsqueeze(0) for l in label], dim=0).cuda(self.output_device)
                        # label_language_new = []
                        # for i, part_name in enumerate(["al","head", "hand", "arm", "hip", "leg", "foot"]):
                        #     label_language_new.append(self.action_descriptions[i])
                        # label_language_new = torch.cat(label_language_new, dim=1).cuda(self.output_device)
                        # label_language = label_language_new[unseen_classes]
                        # sample_label_language = torch.cat([label_language_new[l.item()].unsqueeze(0) for l in label], dim=0).cuda(self.output_device)
                        part_language_seen = part_language1[seen_classes]
                        part_language_unseen = part_language1[unseen_classes]
                        # label_language = torch.cat(label_language, dim=1).cuda(self.output_device)[unseen_classes]
                        # label_language = F.normalize(label_language, dim=1)
                        true_label_array = torch.tensor([test_zsl_label_dict[l.item()] for l in label]).cuda(self.output_device)
                        # part_visual_feature, global_visual_feature = self.model(data, self.attribute_features_dict)
                        part_visual_feature, part_visual_feature_pd, global_visual_feature, part_reconstruction_embedding, part_mu_feature, part_logvar_feature, sim_score, memory_weights, class_prob,sample_label_language,part_des_mapping_feature, gcn_feature, gcn_global, ske_feature, global_semantic = self.model(data, self.attribute_features_dict, part_language, sample_label_language,0, part_language_seen)
                        # loss_part_infonce, loss_global_ce = self.model.loss_cal(part_visual_feature, global_visual_feature, part_language, label_language, true_label_array)
                        # loss = loss_part_infonce +loss_global_ce
                        global_vl_pred_idx, true_label_list = self.model.get_zsl_acc(global_visual_feature, part_visual_feature, label_language,part_language_unseen, true_label_array, unseen_classes, part_des_mapping_feature, ske_feature, global_semantic, gcn_global, gcn_feature)
                    if ln == 'test_gzsl':
                        label_language = self.action_descriptions[0].cuda(self.output_device)[:51,:]
                        sample_label_language = torch.cat([self.action_descriptions[0][l.item()].unsqueeze(0) for l in label], dim=0).cuda(self.output_device)
                        # label_language_new = []
                        # for i, part_name in enumerate(["al","head", "hand", "arm", "hip", "leg", "foot"]):
                        #     label_language_new.append(self.action_descriptions[i])
                        # label_language_new = torch.cat(label_language_new, dim=1).cuda(self.output_device)
                        # label_language = label_language_new[:60,:]
                        # sample_label_language = torch.cat([label_language_new[l.item()].unsqueeze(0) for l in label], dim=0).cuda(self.output_device)
                        part_language_unseen_seen = part_language1[:51]
                        part_language_seen = part_language1[:51,:]
                        # label_language = torch.cat(label_language, dim=1).cuda(self.output_device)[:60,:]
                        # label_language = F.normalize(label_language, dim=1)
                        true_label_array = torch.tensor([test_gzsl_label_dict[l.item()] for l in label]).cuda(self.output_device)
                        # part_visual_feature, global_visual_feature = self.model(data, self.attribute_features_dict)
                        part_visual_feature, part_visual_feature_pd, global_visual_feature, part_reconstruction_embedding, part_mu_feature, part_logvar_feature, sim_score, memory_weights, class_prob, sample_label_language,part_des_mapping_feature, gcn_feature, gcn_global, ske_feature, global_semantic = self.model(data, self.attribute_features_dict, part_language, sample_label_language, 0, part_language_seen)
                        # loss_part_infonce, loss_global_ce = self.model.loss_cal(part_visual_feature, global_visual_feature, part_language, label_language, true_label_array)
                        # loss = loss_part_infonce +loss_global_ce
                        (global_vl_pred_idx, true_label_list,threshold_seen_list, threshold_unseen_list, global_vl_pred) = self.model.get_gzsl_acc(global_visual_feature, part_visual_feature,label_language, part_language_unseen_seen,true_label_array, unseen_classes, sim_score,part_mu_feature, part_logvar_feature, part_des_mapping_feature,ske_feature,global_semantic,gcn_global, gcn_feature)
                        threshold_unseen_value += threshold_unseen_list
                        threshold_seen_value += threshold_seen_list
                        sim_score_list.append(sim_score)
                        sim_matrix_list.append(global_vl_pred)
                        class_prob_list.append(class_prob)
                        
                       
                    
                       
                  
                    pred_list.append(global_vl_pred_idx)
                    label_list.append(true_label_list)
                    step += 1

                
            label_list_acc = np.concatenate(label_list)
            pred_list_acc = np.concatenate(pred_list)

            if ln == 'test_zsl':
                acc_list = list(map(lambda x, y: int(x)==int(y), label_list_acc, pred_list_acc))
                accuracy = np.sum(np.array(acc_list))/len(acc_list)
                
                if self.arg.phase == 'train':
                    # self.val_writer.add_scalar('loss', loss, self.global_step)
                    self.val_writer.add_scalar('acc', accuracy, self.global_step)
                # self.print_log('\tMean {} loss of {} batches: {}.'.format(
                # ln, len(self.data_loader[ln]), np.mean(loss_value)))
                self.print_log('\tTop{}: {:.2f}%'.format(
                1, accuracy*100))
            if ln == 'test_gzsl':
                acc_seen_list = []
                acc_unseen_list = []
                tmp_print = []
                for tl, pl in zip(label_list_acc.tolist(), pred_list_acc.tolist()):
                    if tl in unseen_classes:
                        acc_unseen_list.append(int(tl)==int(pl))
                        tmp_print.append(pl)
                    else:
                        acc_seen_list.append(int(tl)==int(pl))

                # re_loss = torch.sum(torch.cat(sim_score_list, dim=0), dim=1).cuda(self.output_device)
                re_loss = []
                sim_matrix = torch.cat(sim_matrix_list,dim=0).cuda(self.output_device)
                # class_prob = torch.cat(class_prob_list, dim=0).cuda(self.output_device)
                class_prob = []

                percentage_factor_list = [i/100 for i in range(5, 100, 5)]
                
                acc_result = self.model.get_gzsl_acc4(re_loss, label_list_acc.tolist(), sim_matrix, unseen_classes, [0.45], class_prob,[i/100 for i in range(5, 100, 5)], [0, 0.001, 0.005]+[i/10000 for i in range(100, 160, 10)]+[i/1000 for i in range(20,100,5)] + [0.25, 0.5, 0.75])  # ntu60 xsub 55/5
                
                for calibration_factor, accuracy_unseen, accuracy_seen in acc_result:
                    harmonic_mean_acc = 2*accuracy_seen*accuracy_unseen/(accuracy_seen+accuracy_unseen)
                    # print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
                    if self.arg.phase == 'train':
                        # self.val_writer.add_scalar('loss', loss, self.global_step)
                        self.val_writer.add_scalar('acc', harmonic_mean_acc, self.global_step)
                    # self.print_log('\tMean {} loss of {} batches: {}.'.format(
                    #     ln, len(self.data_loader[ln]), np.mean(loss_value)))
                    # self.print_log('\tReloss Percentage Factor: {:.2f}'.format(percentage_reloss))
                    # self.print_log('\tProb Percentage Factor: {:.2f}'.format(percentage_prob))
                    self.print_log('\tCalibration Factor: {:.8f}'.format(calibration_factor))
                    self.print_log('\tSeen Acc: {:.2f}%'.format(accuracy_seen*100))
                    self.print_log('\tUnseen Acc: {:.2f}%'.format(accuracy_unseen*100))
                    self.print_log('\tHarmonic Mean Acc: {:.2f}%'.format(harmonic_mean_acc*100))
                    # self.print_log('\tSeen Loss Mean: {:.2f}. Seen Loss Var: {:.2f}'.format(np.mean(np.array(threshold_seen_value)), np.var(np.array(threshold_seen_value))))
                    # self.print_log('\tUnseen Loss Mean: {:.2f}. Unseen Loss Var: {:.2f}'.format(np.mean(np.array(threshold_unseen_value)), np.var(np.array(threshold_unseen_value))))
                
            # acc for each class:
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            confusion = confusion_matrix(label_list, pred_list)
            list_diag = np.diag(confusion)
            list_raw_sum = np.sum(confusion, axis=1)
            each_acc = list_diag / list_raw_sum
            with open('{}/epoch{}_{}_each_class_acc.csv'.format(self.arg.work_dir, epoch + 1, ln), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(each_acc)
                writer.writerows(confusion)


    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)}')
            # self.print_log(f'# Text Parameters: {count_parameters(self.model_text_dict)}')
            # self.print_log(f'# RGB Parameters: {count_parameters(self.model_image_dict)}')
            start_epoch = 0
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and (epoch+1) > self.arg.save_epoch

                self.train(epoch, save_model=save_model)
                # zsl
                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test_zsl'])
                # gzsl
                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test_gzsl'])

           
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-'+str(self.best_acc_epoch)+'*'))[0]
            weights = torch.load(weights_path)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.'+k, v.cuda(self.output_device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)

            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')
            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.arg.print_log = True


            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')
    




def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_parser():
    parser = argparse.ArgumentParser(description='LLMs for Action Recognition')
    parser.add_argument('--work-dir', default='./work_dir/temp', help='the work folder for stroing results.')
    parser.add_argument('-model_saved_name', default='')
    parser.add_argument('--config', default='./config/nturgbd-cross-view/default.yaml', help='path to the configuration file')

    # processor
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--save-score', type=str2bool, default=False, help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument('--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument('--log-interval', type=int, default=100, help='the interval for printing messages (#iteration)')
    parser.add_argument('--save-interval', type=int, default=1, help='the interval for storing models (#iteration)')
    parser.add_argument('--save-epoch', type=int, default=30, help='the start epoch to save model (#iteration)')
    parser.add_argument('--eval-interval', type=int, default=5, help='the interval for evaluating models (#iteration)')
    parser.add_argument('--print-log', type=str2bool, default=True, help='print logging or not')
    parser.add_argument('--show-topk', type=int, default=[1], nargs='+', help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument('--num-worker', type=int, default=32, help='the number of worker for data loader')
    parser.add_argument('--train-feeder-args', action=DictAction, default=dict(), help='the arguments of data loader for training')
    parser.add_argument('--test-feeder-zsl-args', action=DictAction, default=dict(), help='the arguments of data loader for test zsl')
    parser.add_argument('--test-feeder-gzsl-args', action=DictAction, default=dict(), help='the arguments of data loader for test gzsl')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model-args', action=DictAction, default=dict(), help='the arguments of model')
    parser.add_argument('--weights', default=None, help='the weights for network initialization')
    parser.add_argument('--text_weights', default=None, help='the weights for network initialization')
    parser.add_argument('--rgb_weights', default=None, help='the weights for network initialization')
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument('--base-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--step', type=int, default=[20, 40, 60], nargs='+', help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument('--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument('--start-epoch', type=int, default=0, help='start training from which epoch')
    parser.add_argument('--num-epoch', type=int, default=80, help='stop training in which epoch')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='weight decay for optimizer')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0)
    parser.add_argument('--loss-alpha1', type=float, default=0.8)
    parser.add_argument('--loss-alpha2', type=float, default=0.8)
    parser.add_argument('--loss-alpha3', type=float, default=0.8)
    parser.add_argument('--te-lr-ratio', type=float, default=1)

    return parser


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()

