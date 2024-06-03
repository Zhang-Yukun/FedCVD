
from algorithm.echo.centralized import CentralizedSemiSGDTrainer, CentralizedSGDResNetTrainer
from fedlab.utils.functional import setup_seed
from fedlab.utils.logger import Logger
from torch.utils.data import DataLoader
from datetime import datetime
import os
import torch
import torch.optim as optim
import torch.nn as nn
from model.resnet import resnet50
from model.unet import unet
from utils.evaluation import MultiLabelEvaluator
from utils.dataloader import get_dataloader, get_echo_dataset
from utils.io import guarantee_path
import json
import argparse
import wandb

parser = argparse.ArgumentParser(description="Standalone training example")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--max_epoch", type=int, default=50)
parser.add_argument("--n_classes", type=int, default=4)
parser.add_argument("--model", type=str, default="unet")
parser.add_argument("--t1", type=int, default=20)
parser.add_argument("--t2", type=int, default=70)
parser.add_argument("--weight", type=float, default=1)
parser.add_argument("--case_name", type=str, default="")
parser.add_argument("--frac", type=float, default=1.0)

if __name__ == "__main__":
    # os.environ["WANDB_API_KEY"] = "911290d6e062fc819c61dcd34c7387e2e2ccc9b1"
    # os.environ["WANDB_MODE"] = "offline"
    args = parser.parse_args()
    setup_seed(args.seed)
    max_epoch = args.max_epoch
    labeled_dataset = get_echo_dataset(
        [
            os.path.join("/data/zyk/data/dataset/ECHO/preprocessed/client1/train.csv")
        ],
        base_path="/data/zyk/data/dataset/ECHO/preprocessed",
        locations=["client1"],
        file_name="records.h5",
        n_classes=args.n_classes
    )
    train_dataset = get_echo_dataset(
        [
            os.path.join("/data/zyk/data/dataset/ECHO/preprocessed/client" + str(i) + "/train.csv") for i in
            range(1, 4)
        ],
        base_path="/data/zyk/data/dataset/ECHO/preprocessed",
        locations=["client" + str(i) for i in range(1, 4)],
        file_name="records.h5",
        n_classes=args.n_classes
    )
    test_datasets = [get_echo_dataset(
        [os.path.join("/data/zyk/data/dataset/ECHO/preprocessed/client" + str(i) + "/test.csv")],
        base_path="/data/zyk/data/dataset/ECHO/preprocessed",
        locations=["client" + str(i)],
        file_name="records.h5",
        n_classes=args.n_classes
    ) for i in range(1, 4)]

    base_path = f"/data/zyk/code/fedmace_benchmark/output/echo/balance/centralized/semi/{args.model}/seed/"

    batch_size = args.batch_size
    lr = args.lr
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    test_loaders = [
        DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False) for test_dataset in test_datasets
    ]
    model = resnet50(num_classes=args.n_classes) if args.model == "resnet50" else unet(n_classes=args.n_classes)
    criterion = nn.CrossEntropyLoss()

    evaluator = MultiLabelEvaluator()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = base_path + timestamp + "/"
    guarantee_path(output_path)
    setting = {
        "dataset": "ECHO",
        "model": args.model,
        "batch_size": batch_size,
        "lr": lr,
        "criterion": "CELoss",
        "max_epoch": max_epoch,
        "seed": args.seed
    }
    with open(output_path + "setting.json", "w") as f:
        f.write(json.dumps(setting))
    logger = Logger(log_name="centralized", log_file=output_path + "logger.log")
    wandb.init(
        project="FedCVD_ECHO",
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
    trainer = CentralizedSemiSGDTrainer(
        model=model,
        labeled_loader=labeled_loader,
        train_loader=train_loader,
        test_loaders=test_loaders,
        lr=lr,
        criterion=criterion,
        evaluator=evaluator,
        max_epoch=max_epoch,
        output_path=output_path,
        num_classes=args.n_classes,
        device=None,
        logger=logger
    )
    trainer.run(evaluator, args.t1, args.t2, args.weight)
    evaluator.save(output_path + "metric.json")
    torch.save({
        "model": trainer.model.state_dict(),
        "optimizer": trainer.optimizer.state_dict()
    },
        output_path + "model.pth"
    )
    # wandb.finish()