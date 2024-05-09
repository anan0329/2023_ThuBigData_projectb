import torch
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as TF
import os
from PIL import Image

def read_image(img_path, squeeze = True):
    image = Image.open(img_path)
    x = TF.to_tensor(image)
    if squeeze:
        x = x.unsqueeze_(0)
    
    return x


data_transforms = {
    'B_traing1': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# data_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'val': transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }

# data_dir = '/TOPIC/ProjectB'
data_dir = 'data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['B_traing1']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['B_traing1']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['B_traing1']}
class_names = image_datasets['B_traing1'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    print(f'{device=}')
