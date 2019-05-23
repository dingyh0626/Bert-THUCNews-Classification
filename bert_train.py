import torch
from model import ClfModel
import torch.nn as nn
from data import get_loader
from pytorch_pretrained_bert import BertAdam
from config import CLF_CONFIG
import numpy as np
import visdom
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
viz = visdom.Visdom()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 2)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(loss, acc, i, window):
    viz.line(
        X=torch.ones((1, 2)).cpu() * i,
        Y=torch.Tensor([loss, acc]).unsqueeze(0).cpu(),
        win=window,
        update='replace' if i == 0 else 'append'
    )

vis_title = 'BERT'
vis_legend = ['Loss', 'Accuracy']
epoch_plot = create_vis_plot('Epoch', 'Loss&Accuracy', vis_title, vis_legend)


parser = argparse.ArgumentParser(description='Text Classification By Nert')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--max_epochs', default=10, type=int,
                    help='Maximum number of epochs')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


if __name__ == '__main__':
    model = ClfModel(**CLF_CONFIG).cuda()
    if args.resume:
        model.load_state_dict(torch.load(args.resume))
    # optimizer = optim.Adam(model.parameters())
    optimizer = BertAdam(model.parameters(), lr=5e-5)
    Loss = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(args.start_epoch, args.max_epochs):
        losses = []
        pred = []
        gt = []
        for i, (x, y) in enumerate(get_loader(batch_size=200)):
            optimizer.zero_grad()
            x = x.cuda()
            y = y.cuda()
            logits = model(x)
            loss = Loss(logits, y)
            loss.backward()
            optimizer.step()
            l = loss.item()
            losses.append(l)
            pred.append(logits)
            gt.append(y)
            # acc = np.mean((logits.argmax(dim=1).cpu().numpy() == y.cpu().numpy()).astype(np.float32))
            if i % 10 == 0:
                print('Epoch: %d, Iter: %d,  Loss: %f' % (epoch, i, l))

        pred = torch.cat(pred, dim=0).argmax(dim=1).cpu().numpy()
        gt = torch.cat(gt, dim=0).cpu().numpy()
        print('Epoch: %d, Loss: %f, Acc: %f' % (epoch, np.mean(losses), np.mean((pred == gt).astype(np.float32))))
        update_vis_plot(np.mean(losses), np.mean((pred == gt).astype(np.float32)), epoch, epoch_plot)
        # with open(os.path.join(args.save_folder, 'model.pkl'), 'wb') as f:
        #     torch.save(model.state_dict(), f)
    torch.save(model.state_dict(), os.path.join(args.save_folder, 'model.pkl'))
