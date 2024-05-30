
from algorithm.ecg.centralized import CentralizedSGDTrainer
from fedlab.utils.functional import setup_seed
from fedlab.utils.logger import Logger
from torch.utils.data import DataLoader
from datetime import datetime
import os
import torch
import torch.optim as optim
import torch.nn as nn
from model.resnet import resnet1d34
from utils.evaluation import MultiLabelEvaluator
from utils.dataloader import get_dataloader, get_dataset
from utils.io import guarantee_path
import json
import argparse

parser = argparse.ArgumentParser(description="Standalone training example")
parser.add_argument("--seed", type=int, default=42)

if __name__ == "__main__":
    args = parser.parse_args()
    setup_seed(args.seed)

    max_epoch = 50
    train_dataset = get_dataset(
        [
            os.path.join("/data/zyk/data/dataset/ECG/preprocessed/client" + str(i) + "/train_valid_20_r.csv") for i in
            range(1, 5)
        ],
        base_path="/data/zyk/data/dataset/ECG/preprocessed",
        locations=["client" + str(i) for i in range(1, 5)],
        file_name="records_20.h5",
        n_classes=20
    )
    test_datasets = [get_dataset(
        [os.path.join("/data/zyk/data/dataset/ECG/preprocessed/client" + str(i) + "/test_20_r.csv")],
        base_path="/data/zyk/data/dataset/ECG/preprocessed",
        locations=["client" + str(i)],
        file_name="records_20.h5",
        n_classes=20
    ) for i in range(1, 5)]

    base_path = "/data/zyk/code/fedmace_benchmark/output/centralized/seed"

    for batch_size in [32]:
        for lr in [0.1]:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
            test_loaders = [
                DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False) for test_dataset in test_datasets
            ]
            model = resnet1d34()
            criterion = nn.BCELoss()
            evaluator = MultiLabelEvaluator()

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_path = base_path + timestamp + "/"
            guarantee_path(output_path)

            setting = {
                "dataset": "ECG",
                "model": "resnet1d34",
                "batch_size": batch_size,
                "lr": lr,
                "criterion": "BCELoss",
                "max_epoch": max_epoch
            }
            with open(output_path + "setting.json", "w") as f:
                f.write(json.dumps(setting))

            logger = Logger(log_name="centralized", log_file=output_path + "logger.log")

            trainer = CentralizedSGDTrainer(
                model=model,
                train_loader=train_loader,
                test_loaders=test_loaders,
                lr=lr,
                criterion=criterion,
                evaluator=evaluator,
                max_epoch=max_epoch,
                output_path=output_path,
                device=None,
                logger=logger
            )
            trainer.run(evaluator)
            evaluator.save(output_path + "metric.json")
            torch.save({
                "model": trainer.model.state_dict(),
                "optimizer": trainer.optimizer.state_dict()
            },
                output_path + "model.pth"
            )
