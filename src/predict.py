import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from pathlib import Path

from configs import DataConfigs, ResNet34Configs, BeitV2Configs, data_transforms, model_pth

data_configs = DataConfigs()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_ref_table(test_dir: Path | str, has_sub_folder: bool):
    test_dir = Path(test_dir)
    file_cat = []
    if has_sub_folder:
        for subd in test_dir.iterdir():
            label_folder = subd.stem
            for f in subd.iterdir():
                fl = f.name
                file_cat.append((label_folder, fl))
        file_df = pd.DataFrame(file_cat, columns=['label', 'file_name'])
        file_df["file_path"] = test_dir / file_df["label"] / file_df["file_name"]
        file_df["base_path"] = test_dir
    else:
        for f in test_dir.iterdir():
            fl = f.name
            file_cat.append(fl)
        file_df = pd.DataFrame(file_cat, columns=['file_name'])
        file_df["file_path"] = test_dir / file_df["file_name"]
        file_df["base_path"] = test_dir
        
    return file_df[['base_path', 'file_name', 'file_path']]

class ReworkableTestDataset(Dataset):
    def __init__(self, test_ref_df, transform=data_transforms[data_configs.folder_type[-1]]):
        self.image_paths = test_ref_df.file_path
        self.base_path = test_ref_df.base_path[0]
        self.file_names = test_ref_df.file_name
        self.transform = transform
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        x = Image.open(image_path)
        pth = self.file_names[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, pth
    
    def __len__(self):
        return len(self.image_paths)

def predict(test_dataloader, model):
    model = model.to(device)
    model.eval()
    
    files = []
    preds = []
    # trues = []
    
    with torch.inference_mode():
        # for index, (X, path, true_label) in enumerate(test_dataloader):
        for index, (X, path) in enumerate(test_dataloader):
            X = X.to(device)
            pred = model(X)
            pred_labels = pred.argmax(dim=1).cpu().tolist()
            
            files += path
            preds += pred_labels
            # trues += true_label.tolist()
            
    # return pd.DataFrame({"image_name": files, "label": preds, "true_label": trues})
    return pd.DataFrame({"image_name": files, "label": preds})

if __name__ == "__main__":
    
    from configs import SUBMISSION_BASE_FILE_PATH, SUBMISSION_OUT_FILE_PATH
    
    # model 1
    model_path = model_pth["beitv2"]
    target_model_configs = BeitV2Configs()
    model = target_model_configs.model
    model.head = nn.Sequential(
        nn.Linear(in_features=1024, out_features=2), 
        nn.LogSoftmax(dim=1)
    )
    model.load_state_dict(torch.load(model_path))
    
    # model 2
    # model_path = model_pth["resnet34"]
    # target_model_configs = ResNet34Configs()
    # model2 = target_model_configs.model
    # model2.fc = nn.Sequential(
    #     nn.Linear(in_features=512, out_features=2),
    #     nn.LogSoftmax(dim=1)
    # )
    # model2.load_state_dict(torch.load(model_path))

    # model 3
    # model_path = model_pth["beitv2_aug"]
    # target_model_configs = BeitV2Configs()
    # model3 = target_model_configs.model
    # model3.head = nn.Sequential(
    #     nn.Linear(in_features=1024, out_features=2),
    #     nn.LogSoftmax(dim=1)
    # )
    # model3.load_state_dict(torch.load(model_path))

    # model 4
    # mode_path = model_pth["resnet34_aug"]
    # target_model_configs = BeitV2Configs()
    # mode4 = target_model_configs.model
    # mode4.fc = nn.Sequential(
    #     nn.Linear(in_features=512, out_features=2),
    #     nn.LogSoftmax(dim=1)
    # )
    # mode4.load_state_dict(torch.load(model_path))


    data_dir = Path(data_configs.data_dir)
    test_dir = data_configs.folder_type[-1]
    subdir = data_dir / test_dir
    # subdir1 = data_dir / "btest_tmp2"
    
    # test_ref_df = create_ref_table(subdir, has_sub_folder=True)
    test_ref_df = create_ref_table(subdir, has_sub_folder=False)
    val_ds = ReworkableTestDataset(
        test_ref_df, 
        transform=data_transforms[test_dir]
    )
    val_dl = DataLoader(
        val_ds, 
        batch_size=data_configs.batch_size,
        shuffle=False, 
        # num_workers=data_configs.num_workers,
    )
    
    test_df = predict(val_dl, model=model)
    print(test_df.shape)
    print(test_df.head())
    # test_df2 = predict(val_dl, model=model2, prob=True)
    # test_df3 = predict(val_dl, model=model3, prob=True)
    # test_df4 = predict(val_dl, model=model4, prob=True)

    # test_df["label2"] = test_df2["label]'
    # test_df["label3"] = test_df3["label]
    # test_df["label4"] = test_df4["label]
    # test_df["label"] = test_df[['label', 'label2', 'label3', 'label4']].mean(axis=1)


    
    # merge to submissionfile
    submission_df = pd.read_csv(SUBMISSION_BASE_FILE_PATH)
    sub_df = submission_df.drop(columns=['label']).merge(test_df, on="image_name")
    print(sub_df)

    sub_df.to_csv(SUBMISSION_OUT_FILE_PATH, index=False)

