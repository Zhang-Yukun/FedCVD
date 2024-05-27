
from copy import deepcopy
from typing import List
from torch.nn import Module
from torch.utils.data import DataLoader
from fedlab.utils.functional import get_best_device
from fedlab.utils.logger import Logger
from utils.evaluation import Accumulator
from utils.evaluation import transfer_tensor_to_numpy, calculate_accuracy, get_pred_label, calculate_multilabel_metrics
from utils.io import guarantee_path

import torch
import numpy as np
import tqdm
import pandas as pd

class LocalSGDTrainer:
    def __init__(
            self,
            model: Module,
            train_loader: DataLoader,
            test_loaders: List[DataLoader],
            client_idx: int,
            lr: float,
            criterion: Module,
            max_epoch: int,
            output_path: str,
            device: torch.device | None = None,
            logger: Logger | None = None
    ):
        self.train_loader = train_loader
        self.test_loaders = test_loaders
        self.client_idx = client_idx
        self.lr = lr
        self.criterion = criterion
        self.max_epoch = max_epoch
        self._device = get_best_device() if device is None else device
        self._model = deepcopy(model).to(self._device)
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr)
        self._LOGGER = Logger() if logger is None else logger
        self.output_path = output_path
        guarantee_path(self.output_path + "train/")
        guarantee_path(self.output_path + "local_test/")
        guarantee_path(self.output_path + "global_test/")

    def run(self, evaluator):
        for epoch in range(self.max_epoch):
            self.train(epoch, evaluator)
            self.local_test(self.test_loaders[self.client_idx], evaluator, epoch)
            self.global_test(evaluator, epoch)
        torch.save({
            "model": self._model.state_dict()
        },
            self.output_path + "model.pth"
        )
        self._LOGGER.close()

    def train(self, epoch, evaluator):
        self._model.train()
        metric = Accumulator(3)
        pred_score_list = []
        pred_label_list = []
        true_label_list = []
        train_desc = "Epoch {:2d}: train Loss {:.8f}  |  Acc:{:.2f}"
        train_bar = tqdm.tqdm(initial=0, leave=True, total=len(self.train_loader),
                         desc=train_desc.format(epoch, 0, 0), position=0)
        for data, label in self.train_loader:
            data, label = data.to(self._device), label.to(self._device)

            pred_score = self._model(data)
            with torch.no_grad():
                pred_score_np = transfer_tensor_to_numpy(pred_score)
                pred_label_np = transfer_tensor_to_numpy(get_pred_label(pred_score))
                true_label_np = transfer_tensor_to_numpy(label)
                pred_score_list.append(pred_score_np)
                pred_label_list.append(pred_label_np)
                true_label_list.append(true_label_np)

            loss = self.criterion(pred_score, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            metric.add(
                float(loss) * len(label), calculate_accuracy(pred_label_np, true_label_np), len(label)
            )
            train_bar.desc = train_desc.format(epoch, metric[0] / metric[2], metric[1] / metric[2])
            train_bar.update(1)
        train_bar.close()
        all_pred_score_np = np.concatenate(pred_score_list, axis=0)
        all_pred_label_np = np.concatenate(pred_label_list, axis=0)
        all_true_label_np = np.concatenate(true_label_list, axis=0)
        df = pd.DataFrame(all_pred_score_np)
        df.to_csv(
            self.output_path + "train/local_pred_score.csv", index=False, encoding="utf-8"
        )
        df = pd.DataFrame(all_pred_label_np)
        df.to_csv(
            self.output_path + "train/local_pred_label.csv", index=False, encoding="utf-8"
        )
        df = pd.DataFrame(all_true_label_np)
        df.to_csv(
            self.output_path + "train/local_true_label.csv",index=False, encoding="utf-8"
        )
        metric_dict = calculate_multilabel_metrics(all_pred_score_np, all_pred_label_np, all_true_label_np)
        metric_dict["loss"] = metric[0] / metric[2]
        evaluator.add_dict("train", epoch, metric_dict)
        self._LOGGER.info(f"Epoch {epoch} | Train Loss: {metric[0] / metric[2]} | Train Acc: {metric[1] / metric[2]}")

    def local_test(self, test_loader: DataLoader, evaluator, epoch):
        self._model.eval()
        metric = Accumulator(3)
        pred_score_list = []
        pred_label_list = []
        true_label_list = []
        eval_desc = "Local Test Loss {:.8f}  |  Acc:{:.2f}"
        eval_bar = tqdm.tqdm(initial=0, leave=True, total=len(test_loader),
                        desc=eval_desc.format(0, 0), position=0)
        for data, label in test_loader:
            data, label = data.to(self._device), label.to(self._device)
            with torch.no_grad():
                pred_score = self._model(data)

                pred_score_np = transfer_tensor_to_numpy(pred_score)
                pred_label_np = transfer_tensor_to_numpy(get_pred_label(pred_score))
                true_label_np = transfer_tensor_to_numpy(label)

                pred_score_list.append(pred_score_np)
                pred_label_list.append(pred_label_np)
                true_label_list.append(true_label_np)

                loss = self.criterion(pred_score, label)

                metric.add(
                    float(loss) * len(label), calculate_accuracy(pred_label_np, true_label_np), len(label)
                )

            eval_bar.desc = eval_desc.format(metric[0] / metric[2], metric[1] / metric[2])
            eval_bar.update(1)
        eval_bar.close()
        all_pred_score_np = np.concatenate(pred_score_list, axis=0)
        all_pred_label_np = np.concatenate(pred_label_list, axis=0)
        all_true_label_np = np.concatenate(true_label_list, axis=0)
        df = pd.DataFrame(all_pred_score_np)
        df.to_csv(
            self.output_path + "local_test/local_pred_score.csv", index=False, encoding="utf-8"
        )
        df = pd.DataFrame(all_pred_label_np)
        df.to_csv(
            self.output_path + "local_test/local_pred_label.csv", index=False, encoding="utf-8"
        )
        df = pd.DataFrame(all_true_label_np)
        df.to_csv(
            self.output_path + "local_test/local_true_label.csv", index=False, encoding="utf-8"
        )
        metric_dict = calculate_multilabel_metrics(all_pred_score_np, all_pred_label_np, all_true_label_np)
        metric_dict["loss"] = metric[0] / metric[2]
        evaluator.add_dict("local_test", epoch, metric_dict)
        self._LOGGER.info(
            f"Epoch {epoch} | Local Test Loss: {metric[0] / metric[2]} | Local Test Acc: {metric[1] / metric[2]}")

    def global_test(self, evaluator, epoch):
        self._model.eval()
        metric = Accumulator(3)
        pred_score_list = []
        pred_label_list = []
        true_label_list = []
        eval_desc = " Global Test Loss {:.8f}  |  Acc:{:.2f}"
        length = 0
        for item in self.test_loaders:
            length += len(item)
        eval_bar = tqdm.tqdm(initial=0, leave=True, total=length,
                        desc=eval_desc.format(0, 0), position=0)
        for item in self.test_loaders:
            for data, label in item:
                data, label = data.to(self._device), label.to(self._device)
                with torch.no_grad():
                    pred_score = self._model(data)
                    pred_score_np = transfer_tensor_to_numpy(pred_score)
                    pred_label_np = transfer_tensor_to_numpy(get_pred_label(pred_score))
                    true_label_np = transfer_tensor_to_numpy(label)

                    pred_score_list.append(pred_score_np)
                    pred_label_list.append(pred_label_np)
                    true_label_list.append(true_label_np)

                    loss = self.criterion(pred_score, label)

                    metric.add(
                        float(loss) * len(label), calculate_accuracy(pred_label_np, true_label_np), len(label)
                    )

                eval_bar.desc = eval_desc.format(metric[0] / metric[2], metric[1] / metric[2])
                eval_bar.update(1)
        eval_bar.close()
        all_pred_score_np = np.concatenate(pred_score_list, axis=0)
        all_pred_label_np = np.concatenate(pred_label_list, axis=0)
        all_true_label_np = np.concatenate(true_label_list, axis=0)
        df = pd.DataFrame(all_pred_score_np)
        df.to_csv(
            self.output_path + "global_test/local_pred_score.csv", index=False, encoding="utf-8"
        )
        df = pd.DataFrame(all_pred_label_np)
        df.to_csv(
            self.output_path + "global_test/local_pred_label.csv", index=False, encoding="utf-8"
        )
        df = pd.DataFrame(all_true_label_np)
        df.to_csv(
            self.output_path + "global_test/local_true_label.csv", index=False, encoding="utf-8"
        )
        metric_dict = calculate_multilabel_metrics(all_pred_score_np, all_pred_label_np, all_true_label_np)
        metric_dict["loss"] = metric[0] / metric[2]
        evaluator.add_dict("global_test", epoch, metric_dict)
        self._LOGGER.info(
            f"Epoch {epoch} | Global Test Loss: {metric[0] / metric[2]} | Global Test Acc: {metric[1] / metric[2]}")