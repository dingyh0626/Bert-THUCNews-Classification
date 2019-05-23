from torch.utils.data import dataset, DataLoader
import os
import torch
from torch import autograd
from pytorch_pretrained_bert import BertModel, BertTokenizer
import numpy as np
from tqdm import tqdm
from config import DATA_ROOT as ROOT, PROJECT_ROOT as PROOT, MAX_LEN
from .tokenize import pad_sequences
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def __read():
    path_x = os.path.join(ROOT, 'tmp', 'x.npy')
    path_y = os.path.join(ROOT, 'tmp', 'y.npy')
    path_mask = os.path.join(ROOT, 'tmp', 'mask.npy')
    path_trainval_split = os.path.join(ROOT, 'tmp', 'trainval_split.npy')
    x, y, mask, trainval_split = np.load(path_x), np.load(path_y), np.load(path_mask), np.load(path_trainval_split)
    return x, y, mask, trainval_split


def extract_feature(raw_texts):
    if os.path.exists(os.path.join(PROOT, 'chinese_L-12_H-768_A-12', 'pytorch_model.bin')):
        bert_model = BertModel.from_pretrained(os.path.join(PROOT, 'chinese_L-12_H-768_A-12')).cuda()
    else:
        bert_model = BertModel.from_pretrained(os.path.join(PROOT, 'chinese_L-12_H-768_A-12'), from_tf=True)
        torch.save(bert_model.state_dict(), os.path.join(PROOT, 'chinese_L-12_H-768_A-12', 'pytorch_model.bin'))
        bert_model = BertModel.from_pretrained(os.path.join(PROOT, 'chinese_L-12_H-768_A-12')).cuda()

    tokenizer = BertTokenizer.from_pretrained(os.path.join(PROOT, 'chinese_L-12_H-768_A-12/vocab.txt'))
    bert_model.eval()
    if type(raw_texts) is not list:
        raw_texts = [raw_texts]
    ids = []
    for text in raw_texts:
        tokens = tokenizer.tokenize(text)
        ids.append(tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]']))
    x, mask = pad_sequences(ids, MAX_LEN)
    x = torch.LongTensor(x)
    mask = torch.LongTensor(mask)
    with autograd.no_grad():
        _, pooled = bert_model(x.cuda(), attention_mask=mask.cuda(), output_all_encoded_layers=False)
    return pooled


def __extract_feature(train=True, force_new=False):
    path_x = os.path.join(ROOT, 'tmp', 'fea_x.npy')
    path_y = os.path.join(ROOT, 'tmp', 'fea_y.npy')
    path_type = os.path.join(ROOT, 'tmp', 'trainval_split_.npy')
    if not force_new and os.path.exists(path_x) and os.path.exists(path_y) and os.path.exists(path_type):
        x = np.load(path_x)
        y = np.load(path_y)
        trainval_split = np.load(path_type)
        flag = trainval_split if train else ~trainval_split
        x = torch.FloatTensor(x[flag])
        y = torch.LongTensor(y[flag])
    else:
        x, y, mask, trainval_split = __read()
        trainval = torch.LongTensor(trainval_split.astype(np.int32))
        mask = torch.LongTensor(mask)
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        temp = dataset.TensorDataset(x, y, mask, trainval)
        if os.path.exists(os.path.join(PROOT, 'chinese_L-12_H-768_A-12', 'pytorch_model.bin')):
            bert_model = BertModel.from_pretrained(os.path.join(PROOT, 'chinese_L-12_H-768_A-12')).cuda()
        else:
            bert_model = BertModel.from_pretrained(os.path.join(PROOT, 'chinese_L-12_H-768_A-12'), from_tf=True)
            torch.save(bert_model.state_dict(), os.path.join(PROOT, 'chinese_L-12_H-768_A-12', 'pytorch_model.bin'))
            bert_model = BertModel.from_pretrained(os.path.join(PROOT, 'chinese_L-12_H-768_A-12')).cuda()
        bert_model = torch.nn.DataParallel(bert_model)
        bert_model.eval()
        Xs = []
        ys = []
        types = []
        for x, y, m, t in tqdm(DataLoader(temp, batch_size=200, shuffle=False, num_workers=2)):
            with autograd.no_grad():
                _, pooled = bert_model(x.cuda(), attention_mask=m.cuda(), output_all_encoded_layers=False)
            Xs.append(pooled.cpu())
            ys.append(y)
            types.append(t)
        x = torch.cat(Xs, dim=0).numpy()
        y = torch.cat(ys, dim=0).numpy()
        types = torch.cat(types, dim=0).numpy().astype(np.bool)
        np.save(path_x, x)
        np.save(path_y, y)
        np.save(path_type, types)
        flag = types if train else ~types
        x = torch.FloatTensor(x[flag])
        y = torch.LongTensor(y[flag])
    return x, y


__cache = {}


def get_loader(train=True, batch_size=200, force_new=False):
    torch_dataset = __cache.get('train' if train else 'val')
    if torch_dataset:
        return DataLoader(torch_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    torch_dataset = dataset.TensorDataset(*__extract_feature(train, force_new=force_new))
    __cache.setdefault('train' if train else 'val', torch_dataset)
    return DataLoader(torch_dataset, batch_size=batch_size, shuffle=True, num_workers=2)


if __name__ == '__main__':
    # A bug: the result is not consistent
    bert1 = BertModel.from_pretrained(os.path.join(PROOT, 'chinese_L-12_H-768_A-12'), from_tf=True)
    bert2 = BertModel.from_pretrained(os.path.join(PROOT, 'chinese_L-12_H-768_A-12'), from_tf=True)
    x, _, mask, _ = __read()
    x = torch.LongTensor(x)[:4]
    mask = torch.LongTensor(mask)[:4]
    bert1.eval()
    bert2.eval()
    with autograd.no_grad():
        _, pooled1 = bert1(x, attention_mask=mask, output_all_encoded_layers=False)
        _, pooled2 = bert2(x, attention_mask=mask, output_all_encoded_layers=False)
    print(torch.sum(pooled1 - pooled2))