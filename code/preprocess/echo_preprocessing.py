
import os
import tqdm
import SimpleITK as sitk
import numpy as np
import pandas as pd
import cv2
import skimage.draw
import collections


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


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)


class EchoPreprocessor:
    def __init__(self,
                 input_path: str,
                 output_path: str
                 ):
        self.input_path = input_path
        self.output_path = output_path
        self.meta = {}
        self.file_list = pd.read_csv(os.path.join(input_path, "FileList.csv"))
        frames = collections.defaultdict(list)
        trace = collections.defaultdict(_defaultdict_of_lists)
        with open(os.path.join(input_path, "VolumeTracings.csv"), "r") as file:
            header = file.readline().strip().split(",")
            assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

            for line in file:
                filename, x1, y1, x2, y2, frame = line.strip().split(',')
                x1 = float(x1)
                y1 = float(y1)
                x2 = float(x2)
                y2 = float(y2)
                frame = int(frame)
                if frame not in trace[filename]:
                    frames[filename].append(frame)
                trace[filename][frame].append((x1, y1, x2, y2))
            for filename in frames:
                for frame in frames[filename]:
                    trace[filename][frame] = np.array(trace[filename][frame])
        self.volume_tracing = trace
        self.frames = frames
        self.idx = 1

    def convert(self, file_df, output_path):
        self.meta = {}
        os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "labels"), exist_ok=True)

        for index, row in tqdm.tqdm(file_df.iterrows(), total=len(file_df)):
            video = loadvideo(str(os.path.join(self.input_path, "Videos", row["FileName"] + ".avi")))
            masks = {}
            for frame in self.frames[f"{row['FileName']}.avi"]:
                trace = self.volume_tracing[f"{row['FileName']}.avi"][frame]
                x1, y1, x2, y2 = trace[:, 0], trace[:, 1], trace[:, 2], trace[:, 3]
                x = np.concatenate((x1[1:], np.flip(x2[1:])))
                y = np.concatenate((y1[1:], np.flip(y2[1:])))
                r, c = skimage.draw.polygon(np.rint(y).astype(int), np.rint(x).astype(int),
                                            (video.shape[1], video.shape[2]))
                mask = np.zeros((video.shape[1], video.shape[2]), np.float32)
                mask[r, c] = 1
                masks[frame] = mask

            masks = sorted(masks.items(), key=lambda e: np.sum(e[1]), reverse=True)
            for pos, item in enumerate(masks):
                data = {
                    "file_name": f"patient{self.idx:04d}_4CH_{name_list[pos]}",
                    "ED": masks[0][0],
                    "ES": masks[1][0],
                    "LVedv": row["EDV"], "LVesv": row["ESV"], "LVef": row["EF"],
                    "NbFrame": row["NumberOfFrames"]
                }
                sitk.WriteImage(sitk.GetImageFromArray(video[item[0], :, :]),
                                os.path.join(output_path, "images", data["file_name"] + ".mhd"))
                sitk.WriteImage(sitk.GetImageFromArray(item[1]),
                                os.path.join(output_path, "labels", data["file_name"] + ".mhd"))
                self.meta[data["file_name"]] = data
            self.idx += 1

    def run(self):
        self.convert(
            self.file_list, str(os.path.join(self.output_path, "data"))
        )
        self.meta = pd.DataFrame.from_dict(self.meta, orient="index")
        self.meta.to_csv(os.path.join(self.output_path, "metadata.csv"))


if __name__ == "__main__":
    echoProcessor = EchoPreprocessor(
        "/Users/zhangyukun/Downloads/Datasets/ECHO/VEDIO/EchoNet-Dynamic",
        "/Users/zhangyukun/Downloads/Datasets/ECHO/preprocessed/client2/"
    )
    echoProcessor.run()
    # print(is_float("1.2"))
    # with open("/Users/zhangyukun/Downloads/Datasets/CAMUS/training/patient0001/Info_2CH.cfg", "r") as file:
    #     lines = file.readlines()
    #     data = {}
    #     for line in lines:
    #         key, value = line.split(":")
    #         data[key] = value[:-1]

    # config = configparser.ConfigParser()
    # config.read("/Users/zhangyukun/Downloads/Datasets/CAMUS/training/patient0001/Info_2CH.cfg", encoding="utf-8")

