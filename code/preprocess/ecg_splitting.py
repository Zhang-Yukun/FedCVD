
import random
import tqdm
# import hydra
import numpy as np
import pandas as pd
# from omegaconf import DictConfig, OmegaConf


class ECGSplitter:
    def __init__(self,
                 input_file: str,
                 train_output_file: str,
                 test_output_file: str,
                 output_columns_list: list[str, ...],
                 n_classes: int,
                 test_sample_rate: float,
                 random_seed: int,
                 mode: str):
        self.input_file = input_file
        self.meta = None
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.output_columns_list = output_columns_list
        self.train_output_file = train_output_file
        self.test_output_file = test_output_file
        self.n_classes = n_classes
        self.test_sample_rate = test_sample_rate
        self.random_seed = random_seed
        self.mode = mode

    def file_load(self):
        self.meta = pd.read_csv(self.input_file, dtype={'ECG_ID': str})

    def file_save(self):
        self.train_data.sort_values(by=["ECG_ID"], ascending=True, inplace=True)
        # self.valid_data.sort_values(by=["ECG_ID"], ascending=True, inplace=True)
        self.test_data.sort_values(by=["ECG_ID"], ascending=True, inplace=True)

        self.train_data[self.output_columns_list].to_csv(self.train_output_file, index=None, encoding="utf-8")
        # self.valid_data[self.output_columns_list].to_csv(self.valid_output_file, index=None, encoding="utf-8")
        self.test_data[self.output_columns_list].to_csv(self.test_output_file, index=None, encoding="utf-8")

    def label_expand(self):
        for i in range(self.n_classes):
            column_name = str(i)
            self.meta[column_name] = pd.Series(np.zeros(self.meta.shape[0], dtype=int))
        for idx, row in tqdm.tqdm(self.meta.iterrows(), total=self.meta.shape[0]):
            for label in row["Code_Label"].split(";"):
                self.meta.loc[idx, label] = 1

    def ecg_sample(self, data, sample_rate: float):
        if self.mode == "random":
            selected_data = data.sample(frac=sample_rate, random_state=self.random_seed)
        elif self.mode == "hierarchical":
            dataframe_list = []
            for i in range(self.n_classes):
                column_name = str(i)
                dataframe_list.append(
                    data[data[column_name] == 1].sample(frac=sample_rate, random_state=self.random_seed)
                )
            selected_data = pd.concat(dataframe_list)
            selected_data.drop_duplicates(subset=["ECG_ID"], keep="first", inplace=True)
        else:
            raise ValueError("Invalid mode keyword")
        unselected_data = data.drop(labels=selected_data.index)
        return unselected_data, selected_data

    def run(self):
        self.file_load()
        # self.label_expand()
        self.train_data, self.test_data = self.ecg_sample(self.meta, self.test_sample_rate)
        # self.train_data, self.valid_data = self.ecg_sample(self.train_data, self.valid_sample_rate)
        self.file_save()


# @hydra.main(version_base=None, config_path="../configs/ecg/splitting_setting/", config_name="client1_20_r")
# def client1_ecg_split(cfg: DictConfig):
#     ecg_splitter = ECGSplitter(
#         input_file=cfg.splitting_setting.input_file,
#         train_output_file=cfg.splitting_setting.train_output_file,
#         valid_output_file=cfg.splitting_setting.valid_output_file,
#         test_output_file=cfg.splitting_setting.test_output_file,
#         output_columns_list=cfg.splitting_setting.output_columns_list,
#         n_classes=cfg.splitting_setting.n_classes,
#         valid_sample_rate=cfg.splitting_setting.valid_sample_rate,
#         test_sample_rate=cfg.splitting_setting.test_sample_rate,
#         random_seed=cfg.splitting_setting.random_seed,
#         mode=cfg.splitting_setting.mode
#     )
#     ecg_splitter.run()
#
#
# @hydra.main(version_base=None, config_path="../configs/ecg/splitting_setting/", config_name="client2_20_r")
# def client2_ecg_split(cfg: DictConfig):
#     ecg_splitter = ECGSplitter(
#         input_file=cfg.splitting_setting.input_file,
#         train_output_file=cfg.splitting_setting.train_output_file,
#         valid_output_file=cfg.splitting_setting.valid_output_file,
#         test_output_file=cfg.splitting_setting.test_output_file,
#         output_columns_list=cfg.splitting_setting.output_columns_list,
#         n_classes=cfg.splitting_setting.n_classes,
#         valid_sample_rate=cfg.splitting_setting.valid_sample_rate,
#         test_sample_rate=cfg.splitting_setting.test_sample_rate,
#         random_seed=cfg.splitting_setting.random_seed,
#         mode=cfg.splitting_setting.mode
#     )
#     ecg_splitter.run()
#
#
# @hydra.main(version_base=None, config_path="../configs/ecg/splitting_setting/", config_name="client3_20_h")
# def client3_ecg_split(cfg: DictConfig):
#     ecg_splitter = ECGSplitter(
#         input_file=cfg.splitting_setting.input_file,
#         train_output_file=cfg.splitting_setting.train_output_file,
#         valid_output_file=cfg.splitting_setting.valid_output_file,
#         test_output_file=cfg.splitting_setting.test_output_file,
#         output_columns_list=cfg.splitting_setting.output_columns_list,
#         n_classes=cfg.splitting_setting.n_classes,
#         valid_sample_rate=cfg.splitting_setting.valid_sample_rate,
#         test_sample_rate=cfg.splitting_setting.test_sample_rate,
#         random_seed=cfg.splitting_setting.random_seed,
#         mode=cfg.splitting_setting.mode
#     )
#     ecg_splitter.run()
#
#
# @hydra.main(version_base=None, config_path="../configs/ecg/splitting_setting/", config_name="client4_20_h")
# def client4_ecg_split(cfg: DictConfig):
#     ecg_splitter = ECGSplitter(
#         input_file=cfg.splitting_setting.input_file,
#         train_output_file=cfg.splitting_setting.train_output_file,
#         valid_output_file=cfg.splitting_setting.valid_output_file,
#         test_output_file=cfg.splitting_setting.test_output_file,
#         output_columns_list=cfg.splitting_setting.output_columns_list,
#         n_classes=cfg.splitting_setting.n_classes,
#         valid_sample_rate=cfg.splitting_setting.valid_sample_rate,
#         test_sample_rate=cfg.splitting_setting.test_sample_rate,
#         random_seed=cfg.splitting_setting.random_seed,
#         mode=cfg.splitting_setting.mode
#     )
#     ecg_splitter.run()


if __name__ == "__main__":
    splitter = ECGSplitter(
        input_file="/data/zyk/data/dataset/ECG/noniid/client4/metadata.csv",
        train_output_file="/data/zyk/data/dataset/ECG/noniid/client4/train.csv",
        test_output_file="/data/zyk/data/dataset/ECG/noniid/client4/test.csv",
        output_columns_list=["ECG_ID", "Code_Label", "Age", "Sex", "Location"],
        n_classes=20,
        test_sample_rate=0.2,
        random_seed=2,
        mode="random"
    )
    splitter.run()
    # client1_ecg_split()
    # client2_ecg_split()
    # client3_ecg_split()
    # client4_ecg_split()
