import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import timm
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

from configs import (
    DataConfigs, TinyVggConfigs, ResNet34Configs, BeitV2Configs, data_transforms
)
from dataloader import create_dataloaders
from trainer import train
from model import LabelSmoothing
from utils import seed_everything, freeze_pretrained_layers, save_model


data_configs = DataConfigs()

data_dir = Path(data_configs.data_dir)
model_dir = Path(data_configs.model_dir)

seed_everything()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_model(model_arg, class_names):
    if model_arg == "tinyvgg":
        target_model_configs = TinyVggConfigs()
        model = target_model_configs.model(
            input_shape=3,
            hidden_units=target_model_configs.hidden_units,
            output_shape=len(class_names)
        )
    elif model_arg == "resnet34":
        target_model_configs = ResNet34Configs()
        model = target_model_configs.model
        model.load_state_dict(torch.load(target_model_configs.model_path))
        # freeze_pretrained_layers(model)
        model.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=2), 
            nn.LogSoftmax(dim=1)
        )
    elif model_arg == "beitv2":
        target_model_configs = BeitV2Configs()
        model = target_model_configs.model
        model.load_state_dict(torch.load(target_model_configs.model_path))
        freeze_pretrained_layers(model)
        model.head = nn.Sequential(
            nn.Linear(in_features=1024, out_features=2), 
            nn.LogSoftmax(dim=1)
        )
    return model.to(device), target_model_configs

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Project B')
    args.add_argument('-m', '--model', choices=["tinyvgg", "resnet34", "beitv2"],
                      help='name of model backbone')
    args.add_argument('-a', '--aug', action='store_true')
    args.add_argument('-lr', '--learning_rate', default=1e-3, type=float,
                      help='learning rate')
    args.add_argument('-e', '--epoch', default=20, type=int,
                      help='number of epochs')
    
    config = args.parse_args()

    aug_flag = config.aug
    print("aug_flag: ", aug_flag)
    aug_parse = "aug"
    
    if len(data_configs.folder_type) > 2:
        train_folder_type = data_configs.folder_type[:-1]
        aug_folder_type = data_configs.folder_type[-1]
        # test_folder_type = data_configs.folder_type[-1]
        test_folder_type = ""
    else:
        train_folder_type = data_configs.folder_type
        aug_folder_type = data_configs.folder_type[-1]
        test_folder_type = ""

    if not aug_flag:
        aug_folder_type = ""
        aug_parse = ""

    train_dataloader, val_dataloader, test_dataloader, class_names, class_mapping = create_dataloaders(
        data_dir=data_dir, 
        train_val_dirs=train_folder_type, 
        test_dir=test_folder_type, 
        aug_dir=aug_folder_type, 
        transform=data_transforms, 
        train_size=data_configs.train_size, 
        batch_size=data_configs.batch_size, 
        num_workers=data_configs.num_workers
    )
    
    
    model, target_model_configs = get_model(config.model, class_names)
    
    # loss_fn = nn.BCELoss()
    # loss_fn = LabelSmoothing(smoothing=0.1)
    loss_fn = nn.NLLLoss(reduction="sum")
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=1)


    best_model, best_epoch, results, best_val_acc = train(
        model=model, 
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=config.epoch,
        device=device
    )

    save_model(
        model=best_model,
        model_dir=model_dir,
        model_name=f"{target_model_configs.model_name}_{aug_parse}_lr_{config.learning_rate}_epoch_{best_epoch}_valacc_{best_val_acc:.4f}.pth"
    )

