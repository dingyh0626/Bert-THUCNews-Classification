from data import extract_feature
from model import ClfModel
from config import CLF_CONFIG, categories, PROJECT_ROOT
import os
import torch
import numpy as np
from torch import autograd
model = ClfModel(**CLF_CONFIG).cuda()
model.load_state_dict(torch.load('../weights/model.pkl'))
model.eval()
text = [
    open(os.path.join(PROJECT_ROOT, 'test', '体育.txt')).read(),
    open(os.path.join(PROJECT_ROOT, 'test', '财经.txt')).read(),
]
feature = extract_feature(text)
with autograd.no_grad():
    out = model(feature)
pred = np.argmax(out.cpu().numpy(), axis=1)
print(categories[pred[0]])
print(categories[pred[1]])