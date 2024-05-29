import argparse
import json
import os
import sys
from datetime import datetime

import torch.nn as nn
import wandb

from torch.utils.data import DataLoader
sys.path.append(os.path.abspath('../'))
from fedlab.utils.functional import setup_seed
from fedlab.utils.logger import Logger
from algorithm.pipeline import Pipeline
from algorithm.ecg.fedopt_mnist import FedOptMNISTServerHandler, FedOptMNISTSerialClientTrainer
from fedlab.contrib.dataset import PathologicalMNIST
from model.resnet import resnet1d34
from utils.evaluation import FedClientMultiLabelEvaluator, FedServerMultiLabelEvaluator
from utils.io import guarantee_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--server_lr', type=float, default=0.5)
    parser.add_argument('--client_lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--communication_round', type=int, default=10)
    parser.add_argument('--num_clients', type=int, default=4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=1e-2)
    parser.add_argument('--case_name', type=str, default="")
    parser.add_argument('--option', type=str, default="adagrad")
    args = parser.parse_args()
    setup_seed(args.seed)

    max_epoch = args.epochs
    communication_round = args.communication_round
    num_clients = args.num_clients
    sample_ratio = 1

    datasets = PathologicalMNIST(root='../../datasets/mnist/', path="../../datasets/mnist/", num_clients=num_clients,
                                 download=True, preprocess=True)
    train_datasets = [datasets.get_dataset(i, type='train') for i in range(num_clients)]
    test_datasets = [datasets.get_dataset(i, type='test') for i in range(num_clients)]

    base_path = "/data/stupidtree/project/FedCVD/output/fedopt/fedadam/"
    os.makedirs(base_path, exist_ok=True)

    beta1 = args.beta1
    beta2 = args.beta2
    tau = args.tau
    option = args.option
    wandb.init(project='CVD', name=args.case_name)

    # for lr in [0.1]:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = base_path + timestamp + "/"

    train_loaders = [
        DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) for train_dataset in train_datasets
    ]
    test_loaders = [
        DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False) for test_dataset in test_datasets
    ]
    model = resnet1d34(task='multiclass', num_classes=10, input_channels=1)
    criterion = nn.CrossEntropyLoss()
    client_evaluators = [FedClientMultiLabelEvaluator() for _ in range(1, 5)]
    server_evaluator = FedServerMultiLabelEvaluator()

    for idx in range(1, 5):
        guarantee_path(output_path + "client" + str(idx) + "/")
    guarantee_path(output_path + "server/")

    setting = {
        "dataset": "ECG",
        "model": "resnet1d34",
        "batch_size": args.batch_size,
        "client_lr": args.client_lr,
        "server_lr": args.server_lr,
        "beta1": beta1,
        "beta2": beta2,
        "tau": tau,
        "option": option,
        "criterion": "BCELoss",
        "num_clients": num_clients,
        "sample_ratio": sample_ratio,
        "communication_round": communication_round,
        "max_epoch": max_epoch,
    }
    with open(output_path + "setting.json", "w") as f:
        f.write(json.dumps(setting))

    client_loggers = [
        Logger(log_name="client" + str(idx), log_file=output_path + "client" + str(idx) + "/logger.log")
        for idx in range(1, 5)
    ]
    server_logger = Logger(log_name="server", log_file=output_path + "server/logger.log")

    trainer = FedOptMNISTSerialClientTrainer(
        model=model,
        num_clients=num_clients,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        lr=args.client_lr,
        criterion=criterion,
        max_epoch=max_epoch,
        output_path=output_path,
        evaluators=client_evaluators,
        device=None,
        logger=client_loggers
    )

    handler = FedOptMNISTServerHandler(
        lr=args.server_lr,
        beta1=beta1,
        beta2=beta2,
        tau=tau,
        option=option,
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
