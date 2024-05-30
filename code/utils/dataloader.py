import os

import h5py
import numpy
import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader


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


class EchoDataset(Dataset):
    def __init__(self, path='/data/zyk/data/dataset/ECHO/preprocessed/', location='client1', meta_name='train',
                 pad_value=-200, pad_labels=True):
        meta = pd.read_csv(path + f'{location}/{meta_name}.csv', dtype={"ECHO_ID": str})
        file = h5py.File(path + f'{location}/records.h5', 'r')
        # print(meta)
        # for each row in meta, get its ECHO_ID
        self.videos = []
        self.labels = []
        self.ids = []
        self.pad_labels = pad_labels
        for idx, row in meta.iterrows():
            echo_id = row['ECHO_ID']
            client_id = row['Location']
            obj = file[echo_id]
            # get the video data of the ECHO_ID
            video = numpy.array(obj['video'])
            mask = numpy.array(obj['mask'])

            if self.pad_labels:
                # mask is (seq_len, 112,112)
                pad_size = video.shape[1] - mask.shape[0]
                # create tensor of size (pad_size, 112,112) full of -1
                padding = numpy.full((pad_size, 112, 112), pad_value)
                # concatenate mask and padding
                if mask.shape[0] == 2:  # pad in the middle
                    mask = numpy.concatenate((mask[:1, :, :], padding, mask[1:, :, :]), axis=0)
                else:
                    mask = numpy.concatenate((mask, padding), axis=0)
            self.videos.append(video[0])
            self.labels.append(mask)
            self.ids.append(f'{client_id}_{echo_id}')

    def merge(self, data):
        self.videos.extend(data.videos)
        self.labels.extend(data.labels)
        self.ids.extend(data.ids)

    def subset(self, frac):
        n = int(len(self) * frac)
        res_ds = EchoDataset()
        res_ds.videos = self.videos[:n]
        res_ds.labels = self.labels[:n]
        res_ds.ids = self.ids[:n]
        return res_ds

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, item):
        return dict(id=self.ids[item], video=self.videos[item], label=self.labels[item])


class EchoFrameDataset(Dataset):
    def __init__(self, path='/data/zyk/data/dataset/ECHO/preprocessed/', location='client1', meta_name='train',
                 pad_value=-200, ):
        meta = pd.read_csv(path + f'{location}/{meta_name}.csv', dtype={"ECHO_ID": str})
        file = h5py.File(path + f'{location}/records.h5', 'r')
        # print(meta)
        # for each row in meta, get its ECHO_ID
        self.frames = []
        self.labels = []
        self.ids = []
        self.labeled_index = []
        self.has_label = []
        all_index = 0
        for idx, row in meta.iterrows():
            echo_id = row['ECHO_ID']
            client_id = row['Location']
            obj = file[echo_id]
            # get the video data of the ECHO_ID
            video = numpy.array(obj['video'])
            mask = numpy.array(obj['mask'])
            for frame_idx in range(video.shape[1]):
                self.frames.append(video[0, frame_idx, ...])
                if mask.shape[0] == 2:
                    if frame_idx == 0:
                        self.labels.append(mask[0])
                        self.labeled_index.append(all_index)
                        self.has_label.append(True)
                    elif frame_idx == video.shape[1] - 1:
                        self.labels.append(mask[1])
                        self.labeled_index.append(all_index)
                        self.has_label.append(True)
                    else:
                        self.labels.append(torch.tensor(-200))
                        self.has_label.append(False)
                elif mask.shape[0] == video.shape[1]:
                    self.labels.append(mask[frame_idx])
                    self.labeled_index.append(all_index)
                    self.has_label.append(True)
                else:
                    raise ValueError(f'Invalid mask shape: {mask.shape}')
                self.ids.append(f'{client_id}_{echo_id}_{frame_idx}')
                all_index += 1

    def merge(self, data):
        self.frames.extend(data.frames)
        self.labels.extend(data.labels)
        self.labeled_index.extend(data.labeled_index)
        self.has_label.extend(data.has_label)
        self.ids.extend(data.ids)

    def subset(self, frac):
        n = int(len(self) * frac)
        res_ds = EchoFrameDataset()
        res_ds.frames = self.frames[:n]
        res_ds.labels = self.labels[:n]
        res_ds.has_label = self.has_label[:n]

        return res_ds

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, item):
        return dict(id=self.ids[item], frame=self.frames[item], label=self.labels[item], has_label=self.has_label[item])


class EchoFrameCollator:
    def __init__(self, pad_value=-200):
        self.pad_value = pad_value

    def __call__(self, batch):
        labeled_batch = [item for item in batch if item['has_label']]
        unlabeled_batch = [item for item in batch if not item['has_label']]
        unlabeled_data = None
        if len(unlabeled_batch) > 0:
            unlabeled_ids = [item['id'] for item in unlabeled_batch]
            unlabeled_frames = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(item['frame']) for item in unlabeled_batch],
                batch_first=True)
            unlabeled_data = dict(ids=unlabeled_ids, frames=unlabeled_frames)
        labeled_data = None
        if len(labeled_batch) > 0:
            labeled_frames = torch.nn.utils.rnn.pad_sequence([torch.tensor(item['frame']) for item in labeled_batch],
                                                             batch_first=True)
            labeled_ids = [item['id'] for item in labeled_batch]

            labeled_labels = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(item['label']).long() for item in labeled_batch],
                padding_value=self.pad_value, batch_first=True)
            labeled_data = dict(ids=labeled_ids, frames=labeled_frames, labels=labeled_labels)

        return dict(labeled=labeled_data,
                    unlabeled=unlabeled_data)


class EchoCollator:
    def __init__(self, pad_value=-200):
        self.pad_value = pad_value

    def __call__(self, batch):
        ids = [item['id'] for item in batch]
        videos = torch.nn.utils.rnn.pad_sequence([torch.tensor(item['video']) for item in batch], batch_first=True)
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(item['label']).long() for item in batch],
                                                 padding_value=self.pad_value, batch_first=True)
        return dict(ids=ids, videos=videos, labels=labels)


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
