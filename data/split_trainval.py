from glob import glob
import os
import random
import pandas as pd
from config import DATA_ROOT as ROOT, NUM_TRAIN, NUM_VAL, NUM_ALL, categories


if __name__ == '__main__':
    types = ['train'] * NUM_TRAIN + ['val'] * NUM_VAL
    df = None
    for i, cat in enumerate(categories):
        files = glob(os.path.join(ROOT, cat, '*.txt'))
        random.shuffle(files)
        files = files[:NUM_ALL]
        if df is None:
            df = pd.DataFrame([list(item) for item in zip([i] * NUM_ALL, types, files)],
                              columns=['category', 'type', 'file'])
        else:
            df = df.append(pd.DataFrame([list(item) for item in zip([i] * NUM_ALL, types, files)],
                                        columns=['category', 'type', 'file']))
    if not os.path.exists(os.path.join(ROOT, 'tmp')):
        os.mkdir(os.path.join(ROOT, 'tmp'))
    df.to_csv(os.path.join(ROOT, 'tmp', 'split.csv'), index=0)
