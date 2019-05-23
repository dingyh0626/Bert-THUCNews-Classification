import pandas as pd
from pytorch_pretrained_bert import BertTokenizer
from config import DATA_ROOT as ROOT, MAX_LEN
import numpy as np
from tqdm import tqdm
import os






def pad_sequences(seqs, length):
    seq = []
    mask = []
    for s in seqs:
        if len(s) < length:
            seq.append(s + [0] * (length - len(s)))
            mask.append([1] * len(s) + [0] * (length - len(s)))
        elif len(s) > length:
            seq.append(s[:length])
            mask.append([1] * length)
        else:
            seq.append(s)
            mask.append([1] * length)
    return np.stack(seq, axis=0), np.stack(mask, axis=0)


def parse(df, tokenizer):
    tokens = tokenizer.tokenize(open(df['file'], 'r').read())
    ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
    return ids


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    tokenizer.max_len = 20000
    df = pd.read_csv(os.path.join(ROOT, 'tmp', 'split.csv'))
    texts = []
    masks = []
    cats = []
    trainval = []
    for _, content in tqdm([item for item in df.iterrows()]):
        texts.append(parse(content, tokenizer))
        cats.append(content['category'])
        trainval.append(True if content['type'] == 'train' else False)
    ids, mask = pad_sequences(texts, MAX_LEN)
    cats = np.array(cats, dtype=np.int32)
    trainval_split = np.array(trainval, dtype=np.bool)

    np.save(os.path.join(ROOT, 'tmp', 'x.npy'), ids)
    np.save(os.path.join(ROOT, 'tmp', 'y.npy'), cats)
    np.save(os.path.join(ROOT, 'tmp', 'mask.npy'), mask)
    np.save(os.path.join(ROOT, 'tmp', 'trainval_split.npy'), trainval_split)

