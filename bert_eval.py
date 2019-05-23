import torch
from torch import autograd
from model import ClfModel
import argparse
from data import get_loader
from config import CLF_CONFIG
import numpy as np
from sklearn.metrics import f1_score
parser = argparse.ArgumentParser(description='Text Classification By Nert')
parser.add_argument('--model_path', default='weights/model.pkl', type=str,
                    help='Checkpoint state_dict file path')

args = parser.parse_args()
if __name__ == '__main__':
    model = ClfModel(**CLF_CONFIG).cuda()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    pred = []
    gt = []
    for x, y in get_loader(train=False):
        with autograd.no_grad():
            out = model(x.cuda()).cpu()
        pred.append(out)
        gt.append(y)
    pred = torch.cat(pred, dim=0).argmax(dim=1).numpy()
    gt = torch.cat(gt, dim=0).numpy()
    print('Acc: %f, F1: %f' % (np.mean((pred == gt).astype(np.float32)), f1_score(gt, pred, average='macro')))





