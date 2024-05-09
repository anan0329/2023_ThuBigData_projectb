import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from configs import DataConfigs

data_configs = DataConfigs()
data_dir = Path(data_configs.data_dir)

subdirs = [data_dir / subd for subd in data_configs.folder_type]
file_cat = []
for d in subdirs:
    folder = d.stem
    for subd in d.iterdir():
        label_folder = subd.stem
        for f in subd.iterdir():
            fl = f.name
            file_cat.append((folder, label_folder, fl))
file_df = pd.DataFrame(file_cat, columns=['folder_type', 'label', 'file_name'])
file_df["file_path"] = data_dir / file_df["folder_type"] / file_df["label"] / file_df["file_name"]
file_df["base_path"] = data_dir

train, test = train_test_split(file_df, test_size=0.2, random_state=0, stratify=file_df[['label']])
train = train.sort_values(["folder_type", "label"]).reset_index(drop=True)
test = test.sort_values(["folder_type", "label"]).reset_index(drop=True)

train.to_csv("../train_labels.csv", index=False)
test.to_csv("../test_labels.csv", index=False)

