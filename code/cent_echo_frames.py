import argparse
import os
import sys

import torch.nn.utils.rnn
import wandb
from matplotlib import pyplot as plt
from numpy import inf
from torch.optim import Adam
from torch.utils.data import Dataset
from torchmetrics import Dice
from tqdm import tqdm

from code.utils.evaluation import hausdorff_distance

sys.path.append(os.path.abspath('./code/'))
from fedlab.utils.functional import get_best_device, setup_seed
from model.resnet import resnet50
from utils.dataloader import EchoFrameDataset, EchoFrameCollator


def evaluate(epc, model, loader, ignore_index=-200):
    metric = Dice(ignore_index=0, num_classes=4)
    model.eval()
    with torch.no_grad():
        preds = []
        labels = []
        pred_2ds = []
        label_2ds = []
        images = {}
        for batch_idx, all_batch in enumerate(tqdm(loader, desc='evaluating...')):
            batch = all_batch['labeled']
            if batch is None:
                continue
            input = batch['frames'].to(model.device)
            ipt = input.view(-1, 1, input.size(-1), input.size(-1))
            ipt = torch.cat([ipt, ipt, ipt], dim=1)
            ipt = ipt.float() / 255.0
            output = model(ipt)
            label = batch['labels']
            if (batch_idx in [0, len(loader) - 1, len(loader) // 2]):
                images[batch_idx] = _draw_image(output['out'][0], label[0])

            pred = output['out'].permute(0, 2, 3, 1).reshape(-1, 4).detach().cpu()
            pred_2d = output['out'].permute(0, 2, 3, 1).argmax(-1).detach().cpu()
            real_2d = label.detach().cpu()
            label = label.reshape(-1)
            preds.append(pred)
            labels.append(label)
            pred_2ds.append(pred_2d)
            label_2ds.append(real_2d)
        log_dict = {'epoch': epc}
        for k, v in images.items():
            log_dict[f'images_{k}'] = [wandb.Image(v[0], caption='pred'), wandb.Image(v[1], caption='gt')]
        wandb.log(log_dict)

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        preds_2d = torch.cat(pred_2ds, dim=0)
        labels_2d = torch.cat(label_2ds, dim=0)
        # DICE
        preds = preds[labels != ignore_index]
        labels = labels[labels != ignore_index]
        # preds = preds[labels != 0]
        # labels = labels[labels != 0]
        dice = metric(preds, labels)
        # Hausdorff Distance
        if epc % 5 == 0:
            labels_2d[labels_2d == ignore_index] = 0
            try:
                hd = hausdorff_distance(preds_2d.numpy(), labels_2d.numpy())
            except:
                hd = inf
        else:
            hd = inf

    model.train()
    return dice, hd


def _forward(batch, model):
    input = torch.unsqueeze(batch['frames'], 1).to(model.device)
    ipt = input.view(-1, 1, input.size(-1), input.size(-1))
    ipt = torch.cat([ipt, ipt, ipt], dim=1)
    ipt = ipt.float() / 255.0
    output = model(ipt)
    return output


def _draw_image(pred, label):
    img_gt = label.detach().cpu().numpy()
    img_pred = pred.permute(1, 2, 0).argmax(-1).detach().cpu().numpy()
    color_map = plt.cm.get_cmap('tab20', 4)
    img_gt = (color_map(img_gt) * 255).astype('uint8')
    img_pred = (color_map(img_pred) * 255).astype('uint8')
    return img_pred, img_gt


def centralized_training(args):
    epochs = args.epochs
    batch_size = args.batch_size
    ds_train = EchoFrameDataset(meta_name='train', location='client1')  # .subset(0.01)
    ds_test = EchoFrameDataset(meta_name='test', location='client1')  # .subset(0.1)
    for loc in ['client2', 'client3']:
        ds_train.merge(EchoFrameDataset(meta_name='train', location=loc))
        ds_test.merge(EchoFrameDataset(meta_name='test', location=loc))

    train_dataloader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, collate_fn=EchoFrameCollator())
    test_dataloader = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, collate_fn=EchoFrameCollator())

    device = get_best_device()
    model = resnet50(num_classes=4).to(device)
    model.device = device
    optimizer = Adam(model.parameters(), lr=args.lr)

    wandb.init(project='Echo-Centralized', name='Echo-Centralized-Semi-supervised')
    pbar = tqdm(total=len(train_dataloader) * epochs)
    lf = torch.nn.CrossEntropyLoss(ignore_index=-200, reduction='mean')
    ss_pred_logged = 0
    for epc in range(epochs):
        avg_loss = 0
        num_batches = 0
        for all_batch in train_dataloader:
            batch = all_batch['labeled']
            if batch is None:
                pbar.set_postfix(dict(epc=epc))
                continue
            label = batch['labels'].to(model.device)
            ss_idx = [idx for idx, id in enumerate(batch['ids']) if
                      id.startswith('client2') or id.startswith('client3')]
            svd_idx = [idx for idx, id in enumerate(batch['ids']) if
                       id.startswith('client1')]
            batch_ss = dict(frames=batch['frames'][ss_idx])
            unlabeled_indexes = torch.zeros_like(label).bool()
            zr_idx = label[ss_idx] == 0
            unlabeled_indexes[ss_idx] = zr_idx
            unlabeled_indexes = unlabeled_indexes.view(-1)
            if epc < args.ss_start_epc:  # warm-up, 只在client1数据上训练
                batch = dict(frames=batch['frames'][svd_idx])
                label = label[svd_idx]
                if len(svd_idx) == 0:
                    pbar.update()
                    continue
            else:  # semi-supervised
                with torch.no_grad():
                    ss_raw_pred = _forward(batch_ss, model)
                    sample_label = label[ss_idx[0] if len(ss_idx) > 0 else 0]
                    ss_pred = ss_raw_pred['out'].permute(0, 2, 3, 1).argmax(-1)  # (batch_size, 122,122. 1)
                    label[ss_idx][zr_idx] = ss_pred[zr_idx]
                    if len(ss_idx) > 0 and ss_pred_logged < 3:
                        ss_pred_logged += 1
                        # print('ss_pred:', ss_pred.shape, label.shape)
                        plt.imshow(ss_pred[0][zr_idx[0]])
                        img_pred, img_true = _draw_image(ss_raw_pred['out'][ss_idx[0]], sample_label)
                        _, img_replaced = _draw_image(ss_raw_pred['out'][ss_idx[0]], label[ss_idx[0]])
                        wandb.log({f'wa_images_{ss_pred_logged}': [wandb.Image(img_pred, caption='estimate'),
                                                                   wandb.Image(img_true, caption='gt'),
                                                                   wandb.Image(img_replaced, caption='replaced')]})

            output = _forward(batch, model)
            optimizer.zero_grad()
            pred = output['out'].permute(0, 2, 3, 1).reshape(-1, 4)
            label = label.view(-1).to(device)
            if epc < args.ss_start_epc:  # warm-up, 只在client1数据上训练
                loss = lf(pred, label)
            else:
                supervised_loss = lf(pred[unlabeled_indexes != True], label[unlabeled_indexes != True])
                unsupervised_loss = lf(pred[unlabeled_indexes], label[unlabeled_indexes])
                if unsupervised_loss.isnan():
                    unsupervised_loss = 0
                loss = supervised_loss + args.ss_frac * unsupervised_loss

            avg_loss += loss.detach().item()
            num_batches += 1
            loss.backward()
            optimizer.step()
            pbar.set_postfix(dict(epc=epc, loss=loss.item()))
            pbar.update()
        dice, hd = evaluate(epc, model, test_dataloader)
        wandb.log({'epoch': epc, 'eval_dice': dice.item(), 'eval_hd': hd,
                   'train_loss': avg_loss / num_batches})
        print(f'epoch {epc} dice: {dice.item()}, hd: {hd}')
        if (epc + 1) % 5 == 0:
            torch.save(model.state_dict(),
                       f'/data/stupidtree/project/FedCVD/IO/models/echo/Epoch{epc}-Dice{dice.item():.2f}.pth')

    # save the model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--ss_frac', type=float, default=0.2)
    parser.add_argument('--ss_start_epc', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()
    setup_seed(args.seed)
    centralized_training(args)
