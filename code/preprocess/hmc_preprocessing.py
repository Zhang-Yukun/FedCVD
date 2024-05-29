
import os
import tqdm
import SimpleITK as sitk
import numpy as np
import pandas as pd
import cv2
import skimage.draw
import collections
import scipy.io as sio
import openpyxl


name_list = ["ED", "ES"]


def is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def loadvideo(filename: str) -> np.ndarray:
    """Loads a video from a file.

    Args:
        filename (str): filename of video

    Returns:
        A np.ndarray with dimensions (channels=3, frames, height, width). The
        values will be uint8's ranging from 0 to 255.

    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    v = np.zeros((frame_count, frame_height, frame_width), np.uint8)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        v[count, :, :] = frame

    v = v.transpose((0, 1, 2))

    return v.astype(np.float32)


class HMCQUPreprocessor:
    def __init__(self,
                 input_path: str,
                 output_path: str
                 ):
        self.input_path = input_path
        self.output_path = output_path
        self.file_list = pd.read_excel(os.path.join(self.input_path, "A4C.xlsx"))
        self.meta = {}
        self.idx = 1
        self.file_list = self.file_list[self.file_list["LV Wall Ground-truth Segmentation Masks"] == "ü"]

    def convert(self, file_list, input_path, output_path):
        self.meta = {}
        os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "labels"), exist_ok=True)
        video_path = os.path.join(input_path, "HMC-QU", "A4C")
        mask_path = os.path.join(input_path, "LV Ground-truth Segmentation Masks")

        for index, row in tqdm.tqdm(file_list.iterrows(), total=len(self.file_list)):
            video = loadvideo(str(os.path.join(video_path, row["ECHO"] + ".avi")))
            mask = sio.loadmat(str(os.path.join(mask_path, "Mask_" + row["ECHO"] + ".mat")))
            for frame in range(mask["predicted"].shape[0]):
                # case_identifier = f"patient{idx:04d}_4CH_{frame:02d}"
                # self.meta[case_identifier] = {"file_name": case_identifier}
                resized_image = cv2.resize(video[row["One cardiac-cycle frames"] + frame - 1], (224, 224))
                for x in range(mask["predicted"].shape[1]):
                    for y in range(mask["predicted"].shape[2]):
                        if mask["predicted"][frame][x][y] == 1:
                            mask["predicted"][frame][x][y] = 2
                data = {
                    "file_name": f"patient{self.idx:04d}_4CH_{frame:02d}",
                    "frame": row["One cardiac-cycle frames"] + frame,
                }
                sitk.WriteImage(sitk.GetImageFromArray(resized_image),
                                os.path.join(output_path, "images", data["file_name"] + ".mhd")
                                )
                sitk.WriteImage(sitk.GetImageFromArray(mask["predicted"][frame]),
                                os.path.join(output_path, "labels", data["file_name"] + ".mhd")
                                )
                self.meta[data["file_name"]] = data
            self.idx += 1

    def run(self):
        self.convert(
            self.file_list,
            self.input_path,
            str(os.path.join(self.output_path, "data"))
        )
        self.meta = pd.DataFrame.from_dict(self.meta, orient="index")
        self.meta.to_csv(os.path.join(self.output_path, "metadata.csv"))


if __name__ == "__main__":
    echoProcessor = HMCQUPreprocessor(
        "/Users/zhangyukun/Downloads/Datasets/ECHO/archive/",
        "/Users/zhangyukun/Downloads/Datasets/ECHO/preprocessed/client3/"
    )
    echoProcessor.run()
    # data = pd.read_excel(os.path.join("/Users/zhangyukun/Downloads/Datasets/archive/", "A4C.xlsx"))

