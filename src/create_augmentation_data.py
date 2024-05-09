from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
import random
from typing import Sequence
from pathlib import Path

from dataloader import ReworkableDataset

class RotateRightAngleTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

    
def create_augments(data_path, transform, round=5):
    aug_dataset = ReworkableDataset(data_path, transform)
    base_path = Path("../data/aug")
    # base_path = Path(aug_dataset.base_path) / "aug"
    label0, label1 = aug_dataset.labels.unique()
    (base_path / label0).mkdir(parents=True, exist_ok=True)
    (base_path / label1).mkdir(parents=True, exist_ok=True)
    img_num = 0
    for _ in range(round):
        for  index, (img , label) in enumerate(aug_dataset):
            label_name = str(aug_dataset.labels[index])
            file_name = str(img_num) + "_" +  aug_dataset.file_name[index]
            save_image(img, base_path / label_name / file_name)
            if index == len(aug_dataset) - 1:
                break
        img_num += 1
        

if __name__ == "__main__":
    
    data_path = "../train_labels.csv"
    
    data_aug_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.1),
        transforms.GaussianBlur(kernel_size=(5,5), sigma=(0.1, 1.0)),
        transforms.RandomAdjustSharpness(sharpness_factor=1.3, p=0.5),
        RotateRightAngleTransform([90, 180, 270]),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    create_augments(data_path, data_aug_transforms, round=1)
    
