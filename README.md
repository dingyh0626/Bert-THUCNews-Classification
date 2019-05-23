# Bert-THUCNews-Classification
An PyTorch implementation of text classification based on [Bert](https://github.com/google-research/bert).

## 数据准备

- [Bert中文预训练模型](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)

- [THUCNews中文文本分类语料库](http://thuctc.thunlp.org/message)

## 数据预处理

- `python data/split_trainval.py`: 分割训练和测试数据集，共10个分类，每个分类选择5000个训练文本和2000个测试文本。
- `python data/tokenize.py`: 对原始文本进行处理，转换为文本向量。

## 训练和评估

- `python bert_train.py`: 初次训练会对文本使用Bert进行抽取，每条文本转化为768维的向量，并进行保存。抽取得到的特征经过一层线性变换，使用cross entropy进行训练。10个epoch即可达到不错的效果。
- `python bert_eval.py`: 评估测试集。

## 测试

- `python test/test.py`: 测试`test`文件夹内的两个测试文本。

## 参考

- [Bert官方实现](https://github.com/google-research/bert)

- [PyTorch封装接口](https://github.com/huggingface/pytorch-pretrained-BERT)
