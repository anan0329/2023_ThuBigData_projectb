import torch
torch.set_printoptions(threshold=10_000)

from torchvision.io import read_image
from torchvision.transforms import v2
from pathlib import Path

img_folder = Path("data/B_traing1")
img_rew_folder = img_folder / "reworkable"
img_nrew_folder = img_folder / "not reworkable"

rew_files = [x for x in img_rew_folder.glob("*") if x.is_file()]
nrew_files = [x for x in img_nrew_folder.glob("*") if x.is_file()]

rew_img = read_image(str(rew_files[10]))
nrew_img = read_image(str(nrew_files[10]))
print(v2.Resize(size=30)(rew_img))
print(v2.Resize(size=30)(nrew_img))

