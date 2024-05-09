import random
import torch
import numpy as np
import os
from pathlib import Path

from configs import DEFAULT_RANDOM_SEED


def seed_everything(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# seed_everything()

def freeze_pretrained_layers(model):
    '''Freeze all layers except the last layer(fc or classifier)'''
    for param in model.parameters():
            param.requires_grad = False
    #nn.init.xavier_normal_(model.fc.weight)
    #nn.init.zeros_(model.fc.bias)
    #model.fc.weight.requires_grad = True
    #model.fc.bias.requires_grad = True

def save_model(
    model: torch.nn.Module,
    model_dir: str,
    model_name: str
):
    target_dir_path = Path(model_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
