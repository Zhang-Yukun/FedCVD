
from typing import List

from copy import deepcopy
from torch.nn import Module
from torch.utils.data import DataLoader
from ...fedlab.utils.functional import get_best_device
from ...fedlab.utils.logger import Logger
from ...utils.evaluation import Accumulator
from ...utils.evaluation import shield, cal_hd
from ...utils.io import guarantee_path
from torchmetrics import Dice

import torch
import numpy as np
import tqdm
import pandas as pd
import wandb

class LocalSGDTrainer:
    def __init__(
            self,
            model: Module,
            train_loader: DataLoader,
            test_loaders: List[DataLoader],
            client_idx: int,
            lr: float,
            criterion: Module,
            evaluator,
            max_epoch: int,
            output_path: str,
            num_classes: int,
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
        self.evaluator = evaluator
        self.dice_macro = Dice(ignore_index=0, num_classes=num_classes, average="macro").to(self._device)
        self.dice_micro = Dice(ignore_index=0, num_classes=num_classes, average="macro").to(self._device)
        guarantee_path(self.output_path + "train/")
        guarantee_path(self.output_path + "local_test/")
        guarantee_path(self.output_path + "global_test/")

    @property
    def model(self):
        return self._model

    def run(self, evaluator):
        for epoch in range(self.max_epoch):
            self.train(epoch, evaluator)
            self.local_test(self.test_loaders[self.client_idx - 1], evaluator, epoch)
            self.global_test(evaluator, epoch)
        torch.save({
            "model": self._model.state_dict()
        },
            self.output_path + "model.pth"
        )
        self._LOGGER.close()

    def train(self, epoch, evaluator):
        self._model.train()
        metric = Accumulator(5)
        train_desc = "Epoch {:2d}: train Loss {:.8f} | Dice:{:.2f}"
        train_bar = tqdm.tqdm(initial=0, leave=True, total=len(self.train_loader),
                         desc=train_desc.format(epoch, 0, 0), position=0)
        for data, label, mask in self.train_loader:
            data, label = data.to(self._device), label.to(self._device)

            pred_score = self._model(data)

            loss = self.criterion(pred_score, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                pred_label = pred_score.argmax(dim=1)
                shield_pred_label = shield(pred_label, mask)
                micro_dice = self.dice_micro(shield_pred_label, label)
                macro_dice = self.dice_macro(shield_pred_label, label)
                hd = cal_hd(shield_pred_label.detach().cpu().numpy(), label.detach().cpu().numpy(), mask)
            metric.add(
                float(loss) * len(label),
                micro_dice * len(label),
                macro_dice * len(label),
                hd * len(label),
                len(label)
            )
            wandb.log(
                {
                    "train_loss": metric[0] / metric[-1],
                    "train_micro_dice": metric[1] / metric[-1],
                    "train_macro_dice": metric[2] / metric[-1],
                    "train_hd": metric[3] / metric[-1]
                }
            )
            train_bar.desc = train_desc.format(
                epoch, metric[0] / metric[-1], metric[2] / metric[-1]
            )
            train_bar.update(1)
        train_bar.close()
        metric_dict = {
            "loss": metric[0] / metric[-1],
            "micro_dice": metric[1] / metric[-1],
            "macro_dice": metric[2] / metric[-1],
            "hd": metric[3] / metric[-1]
        }
        evaluator.add_dict("train", epoch, metric_dict)
        self._LOGGER.info(f"Epoch {epoch} | Train Loss: {metric[0] / metric[-1]} | Train Dice: {metric[2] / metric[-1]}")

    def local_test(self, test_loader: DataLoader, evaluator, epoch: int):
        self._model.eval()
        metric_dict = {}
        eval_desc = "Local Test Loss {:.8f} | Dice:{:.2f} | HD:{:.2f}"
        metric = Accumulator(5)
        eval_bar = tqdm.tqdm(initial=0, leave=True, total=len(test_loader),
                             desc=eval_desc.format(0, 0, 0), position=0)
        for data, label, mask in test_loader:
            data, label = data.to(self._device), label.to(self._device)
            with torch.no_grad():
                pred_score = self._model(data)
                pred_label = pred_score.argmax(dim=1)
                shield_pred_label = shield(pred_label, mask)
                micro_dice = self.dice_micro(shield_pred_label, label)
                macro_dice = self.dice_macro(shield_pred_label, label)
                hd = cal_hd(shield_pred_label.detach().cpu().numpy(), label.detach().cpu().numpy(), mask)

                loss = self.criterion(pred_score, label)

                metric.add(
                    float(loss) * len(label),
                    micro_dice * len(label),
                    macro_dice * len(label),
                    hd * len(label),
                    len(label)
                )

            eval_bar.desc = eval_desc.format(
                metric[0] / metric[-1], metric[2] / metric[-1], metric[3] / metric[-1]
            )
            eval_bar.update(1)
        eval_bar.close()
        metric_dict = {
            "loss": metric[0] / metric[-1],
            "micro_dice": metric[1] / metric[-1],
            "macro_dice": metric[2] / metric[-1],
            "hd": metric[3] / metric[-1]
        }
        wandb.log(
            {
                "local_test_loss": metric[0] / metric[-1],
                "local_test_micro_dice": metric[1] / metric[-1],
                "local_test_macro_dice": metric[2] / metric[-1],
                "local_test_hd": metric[3] / metric[-1]
            }
        )
        self._LOGGER.info(f"Epoch {epoch} | Local Test Loss: {metric[0] / metric[-1]} | Local Test Dice: {metric[2] / metric[-1]}")
        evaluator.add_dict("local_test", epoch, metric_dict)

    def global_test(self, evaluator, epoch: int):
        self._model.eval()
        metric = Accumulator(5)
        eval_desc = " Global Test Loss {:.8f} | Dice:{:.2f} | HD:{.2f}"
        length = 0
        for item in self.test_loaders:
            length += len(item)
        eval_bar = tqdm.tqdm(initial=0, leave=True, total=length,
                        desc=eval_desc.format(0, 0), position=0)
        for item in self.test_loaders:
            for data, label, mask in item:
                data, label = data.to(self._device), label.to(self._device)
                with torch.no_grad():
                    pred_score = self._model(data)
                    pred_label = pred_score.argmax(dim=1)
                    shield_pred_label = shield(pred_label, mask)
                    micro_dice = self.dice_micro(shield_pred_label, label)
                    macro_dice = self.dice_macro(shield_pred_label, label)
                    hd = cal_hd(shield_pred_label.detach().cpu().numpy(), label.detach().cpu().numpy(), mask)

                    loss = self.criterion(pred_score, label)

                    metric.add(
                        float(loss) * len(label),
                        micro_dice * len(label),
                        macro_dice * len(label),
                        hd * len(label),
                        len(label)
                    )

                eval_bar.desc = eval_desc.format(metric[0] / metric[-1], metric[2] / metric[-1])
                eval_bar.update(1)
        eval_bar.close()
        metric_dict = {
            "loss": metric[0] / metric[-1],
            "micro_dice": metric[1] / metric[-1],
            "macro_dice": metric[2] / metric[-1],
            "hd": metric[3] / metric[-1]
        }
        wandb.log(
            {
                "global_test_loss": metric[0] / metric[-1],
                "global_test_micro_dice": metric[1] / metric[-1],
                "global_test_macro_dice": metric[2] / metric[-1],
                "global_test_hd": metric[3] / metric[-1]
            }
        )
        evaluator.add_dict("global_test", epoch, metric_dict)
        self._LOGGER.info(f"Epoch {epoch} | Global Test Loss: {metric[0] / metric[-1]} | Global Test Dice: {metric[2] / metric[-1]}")