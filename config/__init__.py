import os

DATA_ROOT = os.path.join(os.environ['HOME'], 'dataset', 'THUCNews')
PROJECT_ROOT = os.path.join(os.environ['HOME'], 'debug-python', 'bert')
MAX_LEN = 400

categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
NUM_TRAIN = 5000
NUM_VAL = 2000
NUM_ALL = NUM_TRAIN + NUM_VAL

CLF_CONFIG = {
    'num_classes': 10,
    'hidden_size': 768
}