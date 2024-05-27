import torch
import pandas as pd
import numpy as np
import h5py
import os
import tqdm
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ECGDataset(Dataset):
    def __init__(self, meta: pd.DataFrame, base_path: str, locations: list[str], file_name: str, n_classes: int):
        super(ECGDataset, self).__init__()
        self.meta = meta
        self.n_classes = n_classes
        self.data_dict = {}
        self.data, self.label = [], []
        for location in locations:
            self.data_dict[location] = h5py.File(os.path.join(base_path, location, file_name), "r")
        for idx in tqdm.tqdm(range(len(self.meta))):
            ecg_id = self.meta.loc[idx, "ECG_ID"]
            location = self.meta.loc[idx, "Location"]
            data = np.array(self.data_dict[location][ecg_id], dtype=float)
            label = np.zeros(self.n_classes, dtype=float)
            idx_list = [int(idx) for idx in self.meta.loc[idx, "Code_Label"].split(";")]
            label[idx_list] = 1
            self.data.append(torch.tensor(data, dtype=torch.float32))
            self.label.append(torch.tensor(label, dtype=torch.float32))

    def __getitem__(self, item):
        return self.data[item], self.label[item]
        # ecg_id = self.meta.loc[item, "ECG_ID"]
        # location = self.meta.loc[item, "Location"]
        # data = np.array(self.data_dict[location][ecg_id], dtype=float)
        # label = np.zeros(self.n_classes, dtype=float)
        # idx_list = [int(idx) for idx in self.meta.loc[item, "Code_Label"].split(";")]
        # label[idx_list] = 1
        # return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.meta)

    def _close_hdf5(self):
        # pass
        for location in self.data_dict.keys():
            self.data_dict[location].close()

    def __del__(self):
        if hasattr(self, 'data_dict'):
            self._close_hdf5()



def get_dataset(
        data_list: list,
        base_path: str,
        locations: list,
        file_name: str,
        n_classes: int
) -> torch.utils.data.Dataset:
    meta = pd.concat(
        [pd.read_csv(data, dtype={"ECG_ID": str}) for data in data_list]
    )
    meta.reset_index(inplace=True)
    dataset = ECGDataset(
        meta, base_path, locations, file_name, n_classes
    )
    return dataset


def get_dataloader(
        data_list: list,
        base_path: str,
        locations: list,
        file_name: str,
        n_classes: int,
        batch_size: int,
        shuffle: bool = True
) -> torch.utils.data.DataLoader:
    meta = pd.concat(
        [pd.read_csv(data, dtype={"ECG_ID": str}) for data in data_list]
    )
    meta.reset_index(inplace=True)
    dataset = ECGDataset(
        meta, base_path, locations, file_name, n_classes
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader