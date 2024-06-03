
from algorithm.echo.fedavg import FedAvgServerHandler, FedAvgSerialClientTrainer
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
from model.unet import unet
from utils.evaluation import FedClientMultiLabelEvaluator, FedServerMultiLabelEvaluator
from utils.dataloader import get_dataloader, get_dataset, get_echo_dataset
from utils.io import guarantee_path
import json
import argparse
import wandb

parser = argparse.ArgumentParser(description="Standalone training example")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--communication_round", type=int, default=50)
parser.add_argument("--max_epoch", type=int, default=1)
parser.add_argument("--n_classes", type=int, default=4)
parser.add_argument("--model", type=str, default="unet")
parser.add_argument("--case_name", type=str, default="")
parser.add_argument("--frac", type=float, default=1.0)

if __name__ == "__main__":
    args = parser.parse_args()
    setup_seed(args.seed)

    max_epoch = args.max_epoch
    communication_round = args.communication_round
    batch_size = args.batch_size
    lr = args.lr
    num_clients = 3
    sample_ratio = 1

    train_datasets = [get_echo_dataset(
        [
            os.path.join("/data/zyk/data/dataset/ECHO/preprocessed/client" + str(i) + "/train.csv")
        ],
        base_path="/data/zyk/data/dataset/ECHO/preprocessed",
        locations=["client" + str(i)],
        file_name="records.h5",
        n_classes=4
    ) for i in range(1, 4)]
    test_datasets = [get_echo_dataset(
        [os.path.join("/data/zyk/data/dataset/ECHO/preprocessed/client" + str(i) + "/test.csv")],
        base_path="/data/zyk/data/dataset/ECHO/preprocessed",
        locations=["client" + str(i)],
        file_name="records.h5",
        n_classes=4
    ) for i in range(1, 4)]

    base_path = "/data/zyk/code/fedmace_benchmark/output/echo/balance/fedavg/seed/"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = base_path + timestamp + "/"

    train_loaders = [
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True) for train_dataset in train_datasets
    ]
    test_loaders = [
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False) for test_dataset in test_datasets
    ]
    model = unet()
    criterion = nn.CrossEntropyLoss()
    client_evaluators = [FedClientMultiLabelEvaluator() for _ in range(1, 4)]
    server_evaluator = FedServerMultiLabelEvaluator()

    for idx in range(1, 4):
        guarantee_path(output_path + "client" + str(idx) + "/")
    guarantee_path(output_path + "server/")

    setting = {
        "dataset": "ECHO",
        "model": "unet",
        "batch_size": batch_size,
        "client_lr": lr,
        "criterion": "CELoss",
        "num_clients": num_clients,
        "sample_ratio": sample_ratio,
        "communication_round": communication_round,
        "max_epoch": max_epoch,
        "seed": args.seed
    }
    with open(output_path + "setting.json", "w") as f:
        f.write(json.dumps(setting))

    wandb.init(
        project="FedCVD_ECHO_FL",
        name=args.case_name,
        config={
            "dataset": "ECHO",
            "model": args.model,
            "batch_size": batch_size,
            "lr": lr,
            "criterion": "CELoss",
            "max_epoch": max_epoch,
            "seed": args.seed
        }
    )

    client_loggers = [
        Logger(log_name="client" + str(idx), log_file=output_path + "client" + str(idx) + "/logger.log") for idx in range(1, 4)
    ]
    server_logger = Logger(log_name="server", log_file=output_path + "server/logger.log")

    trainer = FedAvgSerialClientTrainer(
        model=model,
        num_clients=num_clients,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        num_classes=args.n_classes,
        lr=lr,
        criterion=criterion,
        max_epoch=max_epoch,
        output_path=output_path,
        evaluators=client_evaluators,
        device=None,
        logger=client_loggers
    )

    handler = FedAvgServerHandler(
        num_classes=args.n_classes,
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
