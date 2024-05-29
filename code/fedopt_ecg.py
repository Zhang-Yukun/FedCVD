
from algorithm.ecg.fedopt import FedOptServerHandler, FedOptSerialClientTrainer
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
import jso


if __name__ == "__main__":
    setup_seed(42)

    max_epoch = 1
    communication_round = 50
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

    base_path = "/data/zyk/code/fedmace_benchmark/output/fedopt/fedadam/"

    beta1 = 0.9
    beta2 = 0.999
    tau = 1e-3
    option = "adam"
    for batch_size in [32]:
        for server_lr in [0.001, 0.00001, 0.00001]:
            for client_lr in [0.1, 0.01, 0.001, 0.0001]:
            # for lr in [0.1]:
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
                    "client_lr": client_lr,
                    "server_lr": server_lr,
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

                trainer = FedOptSerialClientTrainer(
                    model=model,
                    num_clients=num_clients,
                    train_loaders=train_loaders,
                    test_loaders=test_loaders,
                    lr=client_lr,
                    criterion=criterion,
                    max_epoch=max_epoch,
                    output_path=output_path,
                    evaluators=client_evaluators,
                    device=None,
                    logger=client_loggers
                )

                handler = FedOptServerHandler(
                    lr=server_lr,
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
