from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from typing import List, Dict
import pandas as pd
from PIL import Image

class ReworkableDataset(Dataset):
    def __init__(self, data_dir, transform):
        data = pd.read_csv(data_dir)
        self.image_paths = data.file_path
        self.base_path = data.base_path[0]
        self.file_name = data.file_name
        self.labels = data.label
        self.transform = transform
        self.classes = sorted(set(self.labels))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        x = Image.open(image_path)
        y = self.labels[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.class_to_idx[y]

    def __len__(self):
        return len(self.image_paths)

def create_dataloaders(
    data_dir: str, 
    train_val_dirs: List[str], 
    test_dir: str, 
    aug_dir: str, 
    transform: Dict[str, transforms.Compose], 
    train_size: float, 
    batch_size: int, 
    num_workers: int
):
    # image_datasets = [ImageFolder(data_dir / x, transform[x])
    #               for x in train_val_dirs]
    # if len(image_datasets) > 1:
    #     train_val_datsets = ConcatDataset(image_datasets)
    # else:
    #     train_val_datsets = image_datasets[0]
    
    # class_names = image_datasets[0].classes
    # class_mapping = image_datasets[0].class_to_idx
    
    # train_size = int(train_size * len(train_val_datsets))
    # val_size = len(train_val_datsets) - train_size
    # train_dataset, val_dataset = random_split(
    #     train_val_datsets, [train_size, val_size])
    
    train_dataset = ReworkableDataset("../train_labels.csv", transform[train_val_dirs[0]])
    val_dataset = ReworkableDataset("../test_labels.csv", transform[train_val_dirs[0]])
    
    class_names = train_dataset.classes
    class_mapping = train_dataset.class_to_idx
    print(class_names)
    print(class_mapping)
    
    if aug_dir:
        image_aug_datasets = ImageFolder(data_dir / "aug", transform["aug"])
        train_dataset = ConcatDataset([train_dataset, image_aug_datasets])
    print(f"{len(train_dataset)=}")
    print(f"{len(val_dataset)=}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_dataloader = None
    
    if test_dir:
        test_datasets = ImageFolder(data_dir / test_dir, 
                                transform[test_dir])
        test_dataloader = DataLoader(
            test_datasets,
            batch_size=batch_size,
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True,
        )
    
    return train_dataloader, val_dataloader, test_dataloader, class_names, class_mapping

if __name__ == "__main__":
    from pathlib import Path
    from configs import DataConfigs, data_transforms

    data_configs = DataConfigs()
    
    data_dir = Path(data_configs.data_dir)
    
    train_folder_type = data_configs.folder_type[:-1]
    aug_folder_type = data_configs.folder_type[-2]
    test_folder_type = ""
    
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
    
    print(class_mapping)
    assert class_mapping == {'not reworkable': 0, 'reworkable': 1}

