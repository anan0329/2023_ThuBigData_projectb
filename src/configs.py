import os
import torch
from torchvision import transforms
from dataclasses import dataclass
from model import TinyVGG, resnet_34, beitv2_224

DEFAULT_RANDOM_SEED = 2023

@dataclass
class DataConfigs:
    # data_dir = '/TOPIC/ProjectB'
    data_dir = '../data'
    model_dir = '../model'
    # folder_type = ['B_traing1', 'B_traing2', 'aug'] # 
    folder_type = ['B_traing1', 'B_traing2', 'aug', 'B_testing'] 
    
    train_size = 0.2
    batch_size = 256
    num_workers = min(os.cpu_count(), 4)
    

@dataclass
class TinyVggConfigs:
    model = TinyVGG
    hidden_units = 10
    model_name = "baseline_tinyvgg_model"


@dataclass
class ResNet34Configs:
    model = resnet_34
    # model_path = "/FTP/timm_model_resnet34.pth"
    model_path = "model/timm_model_resnet34.pth"
    model_name = "resnet34_model"
    

@dataclass
class BeitV2Configs:
    model = beitv2_224
    model_path = "/FTP/timm_model_beitv2_large_patch16_224_in1k_ft_in22k_in1k.pth"
    model_path = "/FTP/timm_model_beitv2_large_patch16_224_in1k_ft_in22k_in1k.pth"
    model_name = "beitv2_large_patch16_224_in1k_ft_in22k_in1k_model"


data_transforms = {
    DataConfigs.folder_type[0]: transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    #DataConfigs.folder_type[1]: transforms.Compose([
    #    transforms.Resize(224),
    #    # transforms.RandomResizedCrop(224),
    #    transforms.RandomHorizontalFlip(),
    #    transforms.ToTensor(),
    #    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #]),
    "aug": transforms.Compose([
    #    transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    DataConfigs.folder_type[-1]: transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

model_pth = {
    "tibyvgg": "../model/baseline_tinyvgg_model_lr_0.001_epoch_20.pth", 
    "resnet34": "../model/resnet34_model__lr_0.0005_epoch_18_valacc_0.97.pth", 
    "resnet34_aug": "../model/resnet34_model_lr_0.001_epoch_18.pth", 
    "beitv2": "../model/beitv2_large_patch16_224_in1k_ft_in22k_in1k_model__lr_0.001_epoch_17_valacc_0.9546.pth", 
    "beitv2_aug": "../model/beitv2_large_patch16_224_in1k_ft_in22k_in1k_model_aug_lr_0.001_epoch_13_valacc_0.9665.pth"
}
