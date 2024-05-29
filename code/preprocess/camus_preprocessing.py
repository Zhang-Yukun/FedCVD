
import os
import tqdm
import SimpleITK as sitk
import numpy as np
import pandas as pd
import cv2 as cv
import h5py


def is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


# class CamusPreprocessor:
#     def __init__(self,
#                  input_path: str,
#                  output_path: str,
#                  sequence: bool,
#                  view: str,
#                  resize: tuple[int, int] = (112, 112)
#                  ):
#         self.input_path = input_path
#         self.output_path = output_path
#         self.sequence = sequence
#         self.view = view
#         self.meta = {}
#         self.idx = 1
#         self.resize = resize
#
#     def convert(self, input_path, output_path):
#         os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
#         os.makedirs(os.path.join(output_path, "labels"), exist_ok=True)
#
#         for case in tqdm.tqdm(sorted(os.listdir(input_path))):
#             case_path = os.path.join(input_path, case)
#             if case == ".DS_Store" or os.path.isfile(case_path):
#                 continue
#             if os.listdir(case_path):
#                 with open(os.path.join(case_path, "Info_" + self.view + ".cfg"), "r") as file:
#                     data = {}
#                     for line in file:
#                         key, value = line.split(":")
#                         value = value[:-1]
#                         if value.isdigit():
#                             value = int(value)
#                         elif is_float(value):
#                             value = float(value)
#                         data[key] = value
#                         data["LargeIndex"] = max((data["ED"], data["ES"]))
#                         data["SmallIndex"] = min((data["ED"], data["ES"]))
#                 case_identifier = f"{case}_{self.view}_sequence"
#                 video = sitk.ReadImage(os.path.join(case_path, f"{case_identifier}.mhd"))
#                 video = sitk.GetArrayFromImage(video)
#                 video = np.array([cv.resize(v, self.resize) for v in video])
#                 f, h, w = video.shape
#                 c = 1
#                 video = video.reshape(c, f, h, w)
#
#                 for instant in ["ED", "ES"]:
#                     case_identifier = f"{case}_{self.view}_{instant}"
#                     image = sitk.ReadImage(os.path.join(case_path, f"{case_identifier}.mhd"))
#                     image = sitk.GetArrayFromImage(image)
#                     if os.path.isfile(os.path.join(case_path, f"{case_identifier}_gt.mhd")):
#                         label = sitk.ReadImage(
#                             os.path.join(case_path, f"{case_identifier}_gt.mhd")
#                         )
#                         label = sitk.GetArrayFromImage(label)
#                     else:
#                         label = None
#                     sitk.WriteImage(
#                         sitk.GetImageFromArray(image[0]),
#                         os.path.join(output_path, "images", f"patient{self.idx:04d}_{self.view}_{instant}.mhd")
#                     )
#                     if label is not None:
#                         sitk.WriteImage(
#                             sitk.GetImageFromArray(label[0]),
#                             os.path.join(output_path, "labels", f"patient{self.idx:04d}_{self.view}_{instant}.mhd")
#                         )
#                     data["file_name"] = str(os.path.join(f"patient{self.idx:04d}_{self.view}_{instant}"))
#                     self.meta[data["file_name"]] = data
#                     self.idx += 1
#
#     def run(self):
#         self.convert(
#             str(os.path.join(self.input_path, "training")), str(os.path.join(self.output_path, "data"))
#         )
#         self.convert(str(os.path.join(self.input_path, "testing")), str(os.path.join(self.output_path, "data")))
#         self.meta = pd.DataFrame.from_dict(self.meta, orient="index")
#         self.meta.to_csv(os.path.join(self.output_path, "metadata.csv"))

class CamusPreprocessor:
    def __init__(self,
                 input_path: str,
                 meta_output_file: str,
                 echo_output_file: str,
                 output_path: str,
                 sequence: bool,
                 view: str,
                 resize: tuple[int, int] = (112, 112)
                 ):
        self.input_path = input_path
        self.meta_output_file = meta_output_file
        self.echo_output_file = echo_output_file
        self.output_path = output_path
        self.sequence = sequence
        self.view = view
        self.meta = {}
        self.idx = 1
        self.resize = resize

    def convert(self, input_path, output_path):
        os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "labels"), exist_ok=True)
        output_file = h5py.File(self.echo_output_file, "w")
        for directory in ["training", "testing"]:
            path = os.path.join(input_path, directory)
            for case in tqdm.tqdm(sorted(os.listdir(path))):
                case_path = os.path.join(path, case)
                if case == ".DS_Store" or os.path.isfile(case_path):
                    continue
                if os.listdir(case_path):
                    with open(os.path.join(case_path, "Info_" + self.view + ".cfg"), "r") as file:
                        data = {}
                        for line in file:
                            key, value = line.split(":")
                            value = value[:-1]
                            if value.isdigit():
                                value = int(value)
                            elif is_float(value):
                                value = float(value)
                            data[key] = value
                            data["LargeIndex"] = max((data["ED"], data["ES"]))
                            data["SmallIndex"] = min((data["ED"], data["ES"]))
                    case_identifier = f"{case}_{self.view}_sequence"
                    video = sitk.ReadImage(os.path.join(case_path, f"{case_identifier}.mhd"))
                    video = sitk.GetArrayFromImage(video)
                    video = np.array([cv.resize(v, self.resize) for v in video])
                    f, h, w = video.shape
                    c = 1
                    video = video.reshape(c, f, h, w)
                    label_dict = {}
                    for instant in ["ED", "ES"]:
                        case_identifier = f"{case}_{self.view}_{instant}"
                        label = sitk.ReadImage(
                            os.path.join(case_path, f"{case_identifier}_gt.mhd")
                        )
                        label_dict[data[instant]] = sitk.GetArrayFromImage(label)
                    output_file[f"echo_{self.idx}"] =
                    self.meta[data["file_name"]] = data
                    self.idx += 1


        for case in tqdm.tqdm(sorted(os.listdir(input_path))):
            case_path = os.path.join(input_path, case)
            if case == ".DS_Store" or os.path.isfile(case_path):
                continue
            if os.listdir(case_path):
                with open(os.path.join(case_path, "Info_" + self.view + ".cfg"), "r") as file:
                    data = {}
                    for line in file:
                        key, value = line.split(":")
                        value = value[:-1]
                        if value.isdigit():
                            value = int(value)
                        elif is_float(value):
                            value = float(value)
                        data[key] = value
                        data["LargeIndex"] = max((data["ED"], data["ES"]))
                        data["SmallIndex"] = min((data["ED"], data["ES"]))
                case_identifier = f"{case}_{self.view}_sequence"
                video = sitk.ReadImage(os.path.join(case_path, f"{case_identifier}.mhd"))
                video = sitk.GetArrayFromImage(video)
                video = np.array([cv.resize(v, self.resize) for v in video])
                f, h, w = video.shape
                c = 1
                video = video.reshape(c, f, h, w)

                for instant in ["ED", "ES"]:
                    case_identifier = f"{case}_{self.view}_{instant}"
                    image = sitk.ReadImage(os.path.join(case_path, f"{case_identifier}.mhd"))
                    image = sitk.GetArrayFromImage(image)
                    if os.path.isfile(os.path.join(case_path, f"{case_identifier}_gt.mhd")):
                        label = sitk.ReadImage(
                            os.path.join(case_path, f"{case_identifier}_gt.mhd")
                        )
                        label = sitk.GetArrayFromImage(label)
                    else:
                        label = None
                    sitk.WriteImage(
                        sitk.GetImageFromArray(image[0]),
                        os.path.join(output_path, "images", f"patient{self.idx:04d}_{self.view}_{instant}.mhd")
                    )
                    if label is not None:
                        sitk.WriteImage(
                            sitk.GetImageFromArray(label[0]),
                            os.path.join(output_path, "labels", f"patient{self.idx:04d}_{self.view}_{instant}.mhd")
                        )
                    data["file_name"] = str(os.path.join(f"patient{self.idx:04d}_{self.view}_{instant}"))
                    self.meta[data["file_name"]] = data
                    self.idx += 1

    def run(self):
        self.convert(
            str(os.path.join(self.input_path, "training")), str(os.path.join(self.output_path, "data"))
        )
        self.convert(str(os.path.join(self.input_path, "testing")), str(os.path.join(self.output_path, "data")))
        self.meta = pd.DataFrame.from_dict(self.meta, orient="index")
        self.meta.to_csv(os.path.join(self.output_path, "metadata.csv"))

if __name__ == "__main__":
    camusProcessor = CamusPreprocessor("/Users/zhangyukun/Downloads/Datasets/ECHO/CAMUS/",
                                       "/Users/zhangyukun/Downloads/Datasets/ECHO/preprocessed/client1/",
                                       False,
                                       "4CH"
                                       )
    camusProcessor.run()
    # print(is_float("1.2"))
    # with open("/Users/zhangyukun/Downloads/Datasets/CAMUS/training/patient0001/Info_2CH.cfg", "r") as file:
    #     lines = file.readlines()
    #     data = {}
    #     for line in lines:
    #         key, value = line.split(":")
    #         data[key] = value[:-1]

    # config = configparser.ConfigParser()
    # config.read("/Users/zhangyukun/Downloads/Datasets/CAMUS/training/patient0001/Info_2CH.cfg", encoding="utf-8")

