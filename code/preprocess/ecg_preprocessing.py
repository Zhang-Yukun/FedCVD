
import wfdb
import hydra
import h5py
import tqdm
import ast
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from abc import ABC, abstractmethod
from pathlib import Path


class Preprocessor:
    def __init__(self,
                 meta_input_file: str,
                 meta_output_file: str,
                 ecg_input_path: str,
                 ecg_output_file: str,
                 location: str,
                 input_columns_list: list[str],
                 output_columns_list: list[str],
                 rename_columns: dict,
                 labels_list: list,
                 labels_map: dict,
                 ecg_length: int,
                 ecg_leads: int
                 ):
        self.meta_input_file = meta_input_file
        self.meta_output_file = meta_output_file
        self.ecg_input_path = ecg_input_path
        self.ecg_output_file = ecg_output_file
        self.location = location
        self.input_columns_list = input_columns_list
        self.output_columns_list = output_columns_list
        self.rename_columns = rename_columns
        self.labels_list = labels_list
        self.labels_map = labels_map
        self.ecg_length = ecg_length
        self.ecg_leads = ecg_leads
        self.meta = None
        self.ecg = None

    def label_load(self):
        self.meta = pd.read_csv(self.meta_input_file)
        self.meta = self.meta[self.input_columns_list]
        self.meta["Method"] = pd.Series(np.zeros(self.meta.shape[0], dtype=int))
        self.meta["Location"] = pd.Series(np.full(self.meta.shape[0], self.location, dtype=object))

    def label_save(self):
        self.meta = self.meta.rename(columns=self.rename_columns)
        self.meta[self.output_columns_list].to_csv(self.meta_output_file, index=False, encoding="utf-8")

    def run(self):
        self.label_load()
        self.label_preprocess()
        self.ecg_preprocess()
        self.label_save()

    def ecg_filter(self, ecg: np.array) -> (np.array, int):
        if ecg.shape[0] == self.ecg_leads:
            ecg = ecg.T
        nan_flag = np.isnan(ecg).any()
        zero_flag = np.all(ecg == 0)
        flag = 1
        if nan_flag or zero_flag:
            flag = 0
        else:
            if ecg.shape[0] < self.ecg_length:
                ecg = self.ecg_padding(ecg)
                flag = 2
            elif ecg.shape[0] > self.ecg_length:
                ecg = self.ecg_cutting(ecg)
                flag = 3
        return ecg.T, flag

    def ecg_padding(self, ecg: np.array) -> np.array:
        padding_col = (0, 0)
        diff = int(self.ecg_length - ecg.shape[0])
        if diff % 2:
            padding_row = (int(diff / 2) + 1, int(diff / 2))
        else:
            padding_row = (int(diff / 2), int(diff / 2))
        ecg = np.pad(ecg, (padding_row, padding_col), mode="edge")
        return ecg

    def ecg_cutting(self, ecg: np.array) -> np.array:
        return ecg[:self.ecg_length][:]

    @abstractmethod
    def label_preprocess(self):
        pass

    @abstractmethod
    def label_split(self, labels):
        pass

    @abstractmethod
    def label_aggregate(self, label_list):
        pass

    @abstractmethod
    def ecg_preprocess(self):
        pass


class SPHPreprocessor(Preprocessor):
    def label_preprocess(self):
        del_rows = []
        for idx, row in tqdm.tqdm(self.meta.iterrows(), total=self.meta.shape[0]):
            label_code = self.label_aggregate(self.label_split(row["AHA_Code"]))
            age_flag = (row["Age"] > 0) and (row["Age"] < 120)
            if label_code and age_flag:
                self.meta.loc[idx, "AHA_Code"] = label_code
            else:
                del_rows.append(idx)
        self.meta.drop(del_rows, inplace=True)
        self.meta.reset_index(inplace=True)

    # 提取 AHA_Code 中主要标签并分割为列表形式
    def label_split(self, labels: str) -> list:
        label_list = []
        # 提取标签
        for labels in labels.split(';'):
            # 筛选主标签
            for label in labels.split('+'):
                x = int(label)
                label_list.append(label)
        label_list = list(set(label_list).intersection(set(self.labels_list)))
        # if "0" in label_list and len(label_list) > 1:
        #     label_list = []
        return label_list

    def label_aggregate(self, label_list: list) -> str:
        map_list = [self.labels_map[label] for label in label_list]
        map_set = set(map_list)
        # if map_set.intersection({"2", "3", "4"}):
        #     map_set.add("1")
        # if map_set.intersection({"11", "12", "13"}):
        #     map_set.add("10")
        map_list = sorted(list(map_set))
        return ";".join(map_list)

    def ecg_preprocess(self):
        output_file = h5py.File(self.ecg_output_file, "w")
        ecg_id = 0
        del_rows = []
        for idx, row in tqdm.tqdm(self.meta.iterrows(), total=self.meta.shape[0]):
            dataset = row["ECG_ID"]
            file = h5py.File(self.ecg_input_path + dataset + ".h5", "r")
            ecg, flag = self.ecg_filter(file["ecg"][:][:])
            if flag != 0:
                output_file[f"{ecg_id:06d}"] = ecg
                self.meta.loc[idx, "ECG_ID"] = f"{ecg_id:06d}"
                self.meta.loc[idx, "Method"] = flag
                ecg_id += 1
            else:
                del_rows.append(idx)
            file.close()
        output_file.close()
        self.meta.drop(del_rows, inplace=True)
        self.meta.reset_index(inplace=True)


class PTBPreprocessor(Preprocessor):
    def label_preprocess(self):
        del_rows = []
        self.meta["scp_codes"] = self.meta["scp_codes"].apply(lambda x: ast.literal_eval(x))
        self.meta["age"] = self.meta["age"].apply(lambda x: int(x))
        for idx, row in tqdm.tqdm(self.meta.iterrows(), total=self.meta.shape[0]):
            label_code = self.label_aggregate(self.label_split(row["scp_codes"]))
            age_flag = (int(row["age"]) > 0) and (int(row["age"]) < 120)
            if label_code and age_flag:
                self.meta.loc[idx, "scp_codes"] = label_code
                self.meta.loc[idx, "age"] = int(self.meta.loc[idx, "age"])
                self.meta.loc[idx, "sex"] = self._sex_map(self.meta.loc[idx, "sex"])
            else:
                del_rows.append(idx)
        self.meta.drop(del_rows, inplace=True)
        self.meta.reset_index(inplace=True)

    def label_split(self, labels: dict) -> list:
        label_list = [key for key in labels.keys()]
        # label_list = [key for key in labels.keys() if float(labels[key]) >= 95.0]
        label_list = list(set(label_list).intersection(set(self.labels_list)))
        # if 'NORM' in label_list and len(label_list) > 1:
        #     label_list = []
        return label_list

    def label_aggregate(self, label_list: list) -> str:
        map_list = [self.labels_map[label] for label in label_list]
        map_list = sorted(list(set(map_list)))
        return ";".join(map_list)

    def ecg_preprocess(self):
        output_file = h5py.File(self.ecg_output_file, "w")
        ecg_id = 0
        del_rows = []
        for idx, row in tqdm.tqdm(self.meta.iterrows(), total=self.meta.shape[0]):
            ecg = [wfdb.rdsamp(self.ecg_input_path + row['filename_hr'])]
            ecg = np.array([signal for signal, meta in ecg], dtype=np.float16)[0]
            ecg, flag = self.ecg_filter(ecg)
            if flag != 0:
                output_file[f"{ecg_id:06d}"] = ecg
                self.meta.loc[idx, "ecg_id"] = f"{ecg_id:06d}"
                self.meta.loc[idx, "Method"] = flag
                ecg_id += 1
            else:
                del_rows.append(idx)
        output_file.close()
        self.meta.drop(del_rows, inplace=True)
        self.meta.reset_index(inplace=True)

    @staticmethod
    def _sex_map(sex: int) -> str:
        sex_map = {0: "M", 1: "F"}
        return sex_map[sex]


class CSNPreprocessor(Preprocessor):
    def label_load(self):
        self.meta = pd.DataFrame({}, columns=self.output_columns_list)

    def label_preprocess(self):
        pass

    def label_split(self, labels: str) -> list[str, ...]:
        label_list = labels.split(",")
        for i in range(len(label_list)):
            label_list[i] = label_list[i].strip()
        label_list = list(set(label_list).intersection(set(self.labels_list)))
        return label_list

    def label_aggregate(self, label_list: list[str, ...]) -> str:
        map_list = [self.labels_map[label] for label in label_list]
        map_list = sorted(list(set(map_list)))
        return ";".join(map_list)

    def ecg_preprocess(self):
        output_file = h5py.File(self.ecg_output_file, "w")
        ecg_id = 0
        del_rows = []
        base_path = Path(self.ecg_input_path)
        first_level_directory_list = sorted(
            [directory for directory in base_path.iterdir() if directory.is_dir()]
        )
        for first_level_directory in tqdm.tqdm(first_level_directory_list):
            second_level_directory_list = sorted(
                [directory for directory in first_level_directory.iterdir() if directory.is_dir()]
            )
            for second_level_directory in second_level_directory_list:
                file_stem_list = sorted(
                    set([path.stem for path in second_level_directory.iterdir()
                         if path.is_file() and path.name != "RECORDS"])
                )
                for file_stem in file_stem_list:
                    ecg, meta = wfdb.rdsamp(str(Path(second_level_directory, file_stem)))
                    comments = {}
                    for item in meta["comments"]:
                        key, value = str(item).split(":")
                        comments[key] = value
                    label_code = self.label_aggregate(self.label_split(comments["Dx"]))
                    if comments["Age"].strip() != "NaN":
                        age = int(comments["Age"])
                    else:
                        age = -1
                    sex = self._sex_map(comments["Sex"])
                    age_flag = ((age > 0) and (age < 120))
                    if label_code and age_flag and sex:
                        ecg = np.array(ecg, dtype=np.float16)
                        ecg, flag = self.ecg_filter(ecg)
                        if flag != 0:
                            output_file[f"{ecg_id:06d}"] = ecg
                            self.meta.loc[ecg_id] = {
                                "ECG_ID": f"{ecg_id:06d}", "Code_Label": label_code, "Age": age,
                                "Sex": sex, "Method": flag, "Location": self.location
                            }
                            ecg_id += 1

    @staticmethod
    def _sex_map(sex: str) -> str | None:
        sex = sex.strip()
        sex_map = {"Male": "M", "Female": "F", "Unknown": None}
        return sex_map[sex]


class GEPreprocessor(Preprocessor):
    def label_load(self):
        self.meta = pd.DataFrame({}, columns=self.output_columns_list)

    def label_preprocess(self):
        pass

    def label_split(self, labels: str) -> list[str, ...]:
        label_list = labels.split(",")
        for i in range(len(label_list)):
            label_list[i] = label_list[i].strip()
        label_list = list(set(label_list).intersection(set(self.labels_list)))
        return label_list

    def label_aggregate(self, label_list: list[str, ...]) -> str:
        map_list = [self.labels_map[label] for label in label_list]
        map_list = sorted(list(set(map_list)))
        return ";".join(map_list)

    def ecg_preprocess(self):
        output_file = h5py.File(self.ecg_output_file, "w")
        ecg_id = 0
        del_rows = []
        base_path = Path(self.ecg_input_path)
        directory_list = sorted(
            [directory for directory in base_path.iterdir() if directory.is_dir()]
        )
        for directory in tqdm.tqdm(directory_list):
            file_stem_list = sorted(
                set([path.stem for path in directory.iterdir()
                     if path.is_file() and path.name != "RECORDS"])
            )
            for file_stem in file_stem_list:
                ecg, meta = wfdb.rdsamp(str(Path(directory, file_stem)))
                comments = {}
                for item in meta["comments"]:
                    key, value = str(item).split(":")
                    comments[key] = value
                label_code = self.label_aggregate(self.label_split(comments["Dx"]))
                if comments["Age"].strip() != "NaN":
                    age = int(comments["Age"])
                else:
                    age = -1
                sex = self._sex_map(comments["Sex"])
                age_flag = ((age > 0) and (age < 120))
                if label_code and age_flag and sex:
                    ecg = np.array(ecg, dtype=np.float16)
                    ecg, flag = self.ecg_filter(ecg)
                    if flag != 0:
                        output_file[f"{ecg_id:06d}"] = ecg
                        self.meta.loc[ecg_id] = {
                            "ECG_ID": f"{ecg_id:06d}", "Code_Label": label_code, "Age": age,
                            "Sex": sex, "Method": flag, "Location": self.location
                        }
                        ecg_id += 1

    @staticmethod
    def _sex_map(sex: str) -> str | None:
        sex = sex.strip()
        sex_map = {"Male": "M", "Female": "F", "Unknown": None}
        return sex_map[sex]


class LUPreprocessor(Preprocessor):
    def label_preprocess(self):
        pass

    def label_split(self, labels):
        pass

    def label_aggregate(self, label_list):
        pass

    def ecg_preprocess(self):
        pass


@hydra.main(version_base=None, config_path="../configs/ecg/preprocessing_setting/", config_name="client1_20")
def sph_preprocess(cfg: DictConfig):
    sph_preprocessor = SPHPreprocessor(
        meta_input_file=cfg.preprocessing_setting.meta_input_file,
        meta_output_file=cfg.preprocessing_setting.meta_output_file,
        ecg_input_path=cfg.preprocessing_setting.ecg_input_path,
        ecg_output_file=cfg.preprocessing_setting.ecg_output_file,
        location=cfg.preprocessing_setting.location,
        input_columns_list=cfg.preprocessing_setting.input_columns_list,
        output_columns_list=cfg.preprocessing_setting.output_columns_list,
        rename_columns=cfg.preprocessing_setting.rename_columns,
        labels_list=cfg.preprocessing_setting.labels_list,
        labels_map=cfg.preprocessing_setting.labels_map,
        ecg_length=cfg.preprocessing_setting.ecg_length,
        ecg_leads=cfg.preprocessing_setting.ecg_leads
    )
    sph_preprocessor.run()


@hydra.main(version_base=None, config_path="../configs/ecg/preprocessing_setting/", config_name="client2_20")
def ptb_preprocess(cfg: DictConfig):
    ptb_preprocessor = PTBPreprocessor(
        meta_input_file=cfg.preprocessing_setting.meta_input_file,
        meta_output_file=cfg.preprocessing_setting.meta_output_file,
        ecg_input_path=cfg.preprocessing_setting.ecg_input_path,
        ecg_output_file=cfg.preprocessing_setting.ecg_output_file,
        location=cfg.preprocessing_setting.location,
        input_columns_list=cfg.preprocessing_setting.input_columns_list,
        output_columns_list=cfg.preprocessing_setting.output_columns_list,
        rename_columns=cfg.preprocessing_setting.rename_columns,
        labels_list=cfg.preprocessing_setting.labels_list,
        labels_map=cfg.preprocessing_setting.labels_map,
        ecg_length=cfg.preprocessing_setting.ecg_length,
        ecg_leads=cfg.preprocessing_setting.ecg_leads
    )
    ptb_preprocessor.run()


@hydra.main(version_base=None, config_path="../configs/ecg/preprocessing_setting/", config_name="client3_20")
def csn_preprocess(cfg: DictConfig):
    csn_preprocessor = CSNPreprocessor(
        meta_input_file=cfg.preprocessing_setting.meta_input_file,
        meta_output_file=cfg.preprocessing_setting.meta_output_file,
        ecg_input_path=cfg.preprocessing_setting.ecg_input_path,
        ecg_output_file=cfg.preprocessing_setting.ecg_output_file,
        location=cfg.preprocessing_setting.location,
        input_columns_list=cfg.preprocessing_setting.input_columns_list,
        output_columns_list=cfg.preprocessing_setting.output_columns_list,
        rename_columns=cfg.preprocessing_setting.rename_columns,
        labels_list=cfg.preprocessing_setting.labels_list,
        labels_map=cfg.preprocessing_setting.labels_map,
        ecg_length=cfg.preprocessing_setting.ecg_length,
        ecg_leads=cfg.preprocessing_setting.ecg_leads
    )
    csn_preprocessor.run()


@hydra.main(version_base=None, config_path="../configs/ecg/preprocessing_setting/", config_name="client4_20")
def ge_preprocess(cfg: DictConfig):
    ge_preprocessor = GEPreprocessor(
        meta_input_file=cfg.preprocessing_setting.meta_input_file,
        meta_output_file=cfg.preprocessing_setting.meta_output_file,
        ecg_input_path=cfg.preprocessing_setting.ecg_input_path,
        ecg_output_file=cfg.preprocessing_setting.ecg_output_file,
        location=cfg.preprocessing_setting.location,
        input_columns_list=cfg.preprocessing_setting.input_columns_list,
        output_columns_list=cfg.preprocessing_setting.output_columns_list,
        rename_columns=cfg.preprocessing_setting.rename_columns,
        labels_list=cfg.preprocessing_setting.labels_list,
        labels_map=cfg.preprocessing_setting.labels_map,
        ecg_length=cfg.preprocessing_setting.ecg_length,
        ecg_leads=cfg.preprocessing_setting.ecg_leads
    )
    ge_preprocessor.run()


if __name__ == "__main__":
    # sph_preprocess()
    # ptb_preprocess()
    # csn_preprocess()
    ge_preprocess()
