
from algorithm.ecg.ditto import DittoServerHandler, DittoSerialClientTrainer
from algorithm.pipeline import Pipeline
from fedlab.utils.functional import setup_seed
from fedlab.utils.logger import Logger
from torch.utils.data import DataLoader
from datetime import datetime
import os
import torch
import torch.optim as optim
import torch.nn as nn
from model.resnet import resnet1d34
from utils.evaluation import FedClientMultiLabelEvaluator, FedServerMultiLabelEvaluator
from utils.dataloader import get_dataloader, get_dataset
from utils.io import guarantee_path
import json
import argparse

parser = argparse.ArgumentParser(description="Standalone training example")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_epoch", type=int, default=1)
parser.add_argument("--communication_round", type=int, default=50)
parser.add_argument("--n_classes", type=int, default=20)
parser.add_argument("--model", type=str, default="resnet1d34")
parser.add_argument("--case_name", type=str, default="")
parser.add_argument("--frac", type=float, default=1.0)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--mu", type=float, default=0.01)


if __name__ == "__main__":
    args = parser.parse_args()
    setup_seed(args.seed)

    max_epoch = args.max_epoch
    communication_round = args.communication_round
    num_clients = 4
    sample_ratio = 1
    train_datasets = [get_dataset(
        [
            os.path.join("/data/zyk/data/dataset/ECG/preprocessed/client" + str(i) + "/train_valid_20_r.csv")
        ],
        base_path="/data/zyk/data/dataset/ECG/preprocessed",
        locations=["client" + str(i)],
        file_name="records_20.h5",
        n_classes=20
    ) for i in range(1, 5)]
    test_datasets = [get_dataset(
        [os.path.join("/data/zyk/data/dataset/ECG/preprocessed/client" + str(i) + "/test_20_r.csv")],
        base_path="/data/zyk/data/dataset/ECG/preprocessed",
        locations=["client" + str(i)],
        file_name="records_20.h5",
        n_classes=20
    ) for i in range(1, 5)]

    base_path = "/data/zyk/code/fedmace_benchmark/output/ditto/seed/"

    batch_size = args.batch_size
    lr = args.lr
    mu = args.mu
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = base_path + timestamp + "/"

    train_loaders = [
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True) for train_dataset in train_datasets
    ]
    test_loaders = [
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False) for test_dataset in test_datasets
    ]
    model = resnet1d34()
    criterion = nn.BCELoss()
    client_evaluators = [FedClientMultiLabelEvaluator() for _ in range(1, 5)]
    server_evaluator = FedServerMultiLabelEvaluator()

    for idx in range(1, 5):
        guarantee_path(output_path + "client" + str(idx) + "/")
    guarantee_path(output_path + "server/")

    setting = {
        "dataset": "ECG",
        "model": "resnet1d34",
        "batch_size": batch_size,
        "client_lr": lr,
        "mu": mu,
        "criterion": "BCELoss",
        "num_clients": num_clients,
        "sample_ratio": sample_ratio,
        "communication_round": communication_round,
        "max_epoch": max_epoch,
        "seed": args.seed
    }
    with open(output_path + "setting.json", "w") as f:
        f.write(json.dumps(setting))

    client_loggers = [
        Logger(log_name="client" + str(idx), log_file=output_path + "client" + str(idx) + "/logger.log")
        for idx in range(1, 5)
    ]
    server_logger = Logger(log_name="server", log_file=output_path + "server/logger.log")

    trainer = DittoSerialClientTrainer(
        mu=mu,
        model=model,
        num_clients=num_clients,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        lr=lr,
        criterion=criterion,
        max_epoch=max_epoch,
        output_path=output_path,
        evaluators=client_evaluators,
        device=None,
        logger=client_loggers
    )

    handler = DittoServerHandler(
        model=model,
        test_loaders=test_loaders,
        criterion=criterion,
        output_path=output_path,
        evaluator=server_evaluator,
        communication_round=communication_round,
        num_clients=num_clients,
        sample_ratio=1,
        device=None,
        logger=server_logger
    )
    standalone = Pipeline(handler, trainer)
    standalone.main()
