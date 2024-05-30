import argparse
import os
import sys

import torch.nn.utils.rnn
import wandb
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.utils.data import Dataset
from torchmetrics import Dice
from tqdm import tqdm

sys.path.append(os.path.abspath('./code/'))
from fedlab.utils.functional import get_best_device, setup_seed
from model.resnet import resnet50
from utils.dataloader import EchoDataset, EchoCollator


def evaluate(epc, model, loader, ignore_index=-200):
    metric = Dice(ignore_index=ignore_index, num_classes=4)
    model.eval()
    with torch.no_grad():
        preds = []
        labels = []
        images = {}
        for batch_idx, batch in enumerate(tqdm(loader, desc='evaluating...')):
            input = batch['videos'].to(model.device)
            for i in range(input.shape[1]):
                ipt = torch.unsqueeze(input[:, i, ...], dim=1)
                ipt = torch.cat([ipt, ipt, ipt], dim=1)
                ipt = ipt.float() / 255.0
                output = model(ipt)
                label = batch['labels'][:, i, ...]
                if (batch_idx in [0, len(loader) - 1, len(loader) // 2]) and i == 0:
                    img_gt = label[0].detach().cpu().numpy()
                    img_pred = output['out'][0].permute(1, 2, 0).argmax(-1).detach().cpu().numpy()
                    color_map = plt.cm.get_cmap('tab20', 4)
                    img_gt = (color_map(img_gt) * 255).astype('uint8')
                    img_pred = (color_map(img_pred) * 255).astype('uint8')
                    images[batch_idx] = (img_gt, img_pred)

                pred = output['out'].permute(0, 2, 3, 1).reshape(-1, 4).detach().cpu()
                label = label.reshape(-1)
                preds.append(pred)
                labels.append(label)
        log_dict = {'epoch': epc}
        for k, v in images.items():
            log_dict[f'images_{k}'] = [wandb.Image(v[0], caption='gt'), wandb.Image(v[1], caption='pred')]
        wandb.log(log_dict)

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        # skip labels where it is -200
        preds = preds[labels != ignore_index]
        labels = labels[labels != ignore_index]
        dice = metric(preds, labels)
    model.train()
    return dice


def centralized_training(args):
    epochs = args.epochs
    batch_size = args.batch_size
    ds_train = EchoDataset(meta_name='train', location='client1')
    ds_test = EchoDataset(meta_name='test', location='client1')
    for loc in ['client2', 'client3']:
        ds_train.merge(EchoDataset(meta_name='train', location=loc))
        ds_test.merge(EchoDataset(meta_name='test', location=loc))

    train_dataloader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, collate_fn=EchoCollator())
    test_dataloader = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, collate_fn=EchoCollator())

    device = get_best_device()
    model = resnet50(num_classes=4).to(device)
    model.device = device
    optimizer = Adam(model.parameters(), lr=args.lr)

    wandb.init(project='Echo-Centralized', name='Echo-Centralized')
    pbar = tqdm(total=len(train_dataloader) * epochs)
    for epc in range(epochs):
        avg_loss = 0
        num_batches = 0
        for batch in train_dataloader:
            input = batch['videos'].to(device)
            loss = 0
            lf = torch.nn.CrossEntropyLoss(ignore_index=-200, reduction='mean')
            optimizer.zero_grad()
            ipt = input.view(-1, 1, input.size(-1), input.size(-1))
            big_batch_size = ipt.size(0)
            print(big_batch_size)
            ipt = torch.cat([ipt, ipt, ipt], dim=1)
            ipt = ipt.float() / 255.0
            output = model(ipt)
            label = batch['labels'].view(-1).to(device)
            pred = output['out'].permute(0, 2, 3, 1).reshape(-1, 4)
            ls = lf(pred, label)
            loss += ls
            # for time_step in range(input.shape[1]):
            #
            #     optimizer.zero_grad()
            #     ipt = torch.unsqueeze(input[:, time_step, ...], dim=1)
            #     ipt = torch.cat([ipt, ipt, ipt], dim=1)
            #     ipt = ipt.float() / 255.0
            #     output = model(ipt)
            #
            #     label = batch['labels'][:, time_step, ...]
            #     pred = output['out'].permute(0, 2, 3, 1).reshape(-1, 4)
            #     label = label.to(device).reshape(-1)
            #     ls = lf(pred, label)
            #
            #
            #     if not ls.isnan():
            #         loss += ls
            avg_loss += loss.detach().item()
            num_batches += 1
            loss.backward()
            optimizer.step()
            pbar.set_postfix(dict(epc=epc, loss=loss.item()))
            pbar.update()
        res = evaluate(epc, model, test_dataloader)
        wandb.log({'epoch': epc, 'eval_dice': res.item(), 'train_loss': avg_loss / num_batches})
        # print(f'epoch {epc} dice: {res.item()}')
        if (epc + 1) % 5 == 0:
            torch.save(model.state_dict(),
                       f'/data/stupidtree/project/FedCVD/IO/models/echo/Epoch{epc}-Dice{res.item():.2f}.pth')

    # save the model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--unlabeled_frac', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()
    setup_seed(args.seed)
    centralized_training(args)
