#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2020/2/9 1:57 下午 
# @Author  : Roger 
# @Version : V 0.1
# @Email   : 550997728@qq.com
# @File    : data_helper.py
import argparse
import os

import numpy as np
import jieba
import re
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.config import root
from utils.multi_proc_utils import parallelize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer


def load_stop_words(stop_word_path):
    '''
    加载停用词
    :param stop_word_path:停用词路径
    :return: 停用词表 list
    '''
    # 打开文件
    file = open(stop_word_path, 'r', encoding='utf-8')
    # 读取所有行
    stop_words = file.readlines()
    # 去除每一个停用词前后 空格 换行符
    stop_words = [stop_word.strip() for stop_word in stop_words]
    return stop_words


def clean_sentence(line):
    line = re.sub(
        "[a-zA-Z0-9]|[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】《》“”！，。？、~@#￥%……&*（）]+|题目", '', line)
    words = jieba.cut(line, cut_all=False)
    return words


stopwords_path = os.path.join(root, 'data', 'stopwords', '哈工大停用词表.txt')
stop_words = load_stop_words(stopwords_path)


def sentence_proc(sentence):
    '''
    预处理模块
    :param sentence:待处理字符串
    :return: 处理后的字符串
    '''
    # 清除无用词
    words = clean_sentence(sentence)
    # 过滤停用词
    words = [word for word in words if word not in stop_words]
    # 拼接成一个字符串,按空格分隔
    return ' '.join(words)


def proc(df):
    df['content'] = df['content'].apply(sentence_proc)
    return df


def data_loader(params, is_rebuild_dataset=False):
    if os.path.exists(os.path.join(root, 'data', 'X_train.npy')) and not is_rebuild_dataset:
        X_train = np.load(os.path.join(root, 'data', 'X_train.npy'))
        X_test = np.load(os.path.join(root, 'data', 'X_test.npy'))
        y_train = np.load(os.path.join(root, 'data', 'y_train.npy'))
        y_test = np.load(os.path.join(root, 'data', 'y_test.npy'))

        with open(os.path.join(params.vocab_save_dir, 'voab.txt'), 'r', encoding='utf-8') as f:
            vocab = {}
            for content in f.readlines():
                k, v = content.strip().split('\t')
                vocab[k] = int(v)
        label_df = pd.read_csv(os.path.join(root, 'data', 'label_baidu95.csv'))
        # 多标签编码
        mlb = MultiLabelBinarizer()
        mlb.fit([label_df['label']])

        return X_train, X_test, y_train, y_test, vocab, mlb

    # 读取数据
    df = pd.read_csv(params.data_path, header=None).rename(columns={0: 'label', 1: 'content'})
    # 并行清理数据
    df = parallelize(df, proc)
    # word2index
    text_preprocesser = Tokenizer(num_words=params.vocab_size, oov_token="<UNK>")
    text_preprocesser.fit_on_texts(df['content'])
    # save vocab
    vocab = text_preprocesser.word_index
    with open(os.path.join(params.vocab_save_dir, 'voab.txt'), 'w', encoding='utf-8') as f:
        for k, v in vocab.items():
            f.write(f'{k}\t{str(v)}\n')

    x = text_preprocesser.texts_to_sequences(df['content'])
    # padding
    x = pad_sequences(x, maxlen=params.padding_size, padding='post', truncating='post')
    # 划分标签
    label_df = pd.read_csv(os.path.join(root, 'data', 'label_baidu95.csv'))
    # 多标签编码
    mlb = MultiLabelBinarizer()
    mlb.fit([label_df['label']])
    df['label'] = df['label'].apply(lambda x: x.split())
    y = mlb.transform(df['label'])
    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # 保存数据
    np.save(os.path.join(root, 'data', 'X_train.npy'), X_train)
    np.save(os.path.join(root, 'data', 'X_test.npy'), X_test)
    np.save(os.path.join(root, 'data', 'y_train.npy'), y_train)
    np.save(os.path.join(root, 'data', 'y_test.npy'), y_test)

    return X_train, X_test, y_train, y_test, vocab, mlb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This is the TextCNN test project.')
    parser.add_argument('-d', '--data_path', default='data/baidu_95.csv', type=str,
                        help='data path')
    parser.add_argument('-v', '--vocab_save_dir', default='data/', type=str,
                        help='data path')
    parser.add_argument('-vocab_size', default=50000, type=int, help='Limit vocab size.(default=50000)')
    parser.add_argument('-p', '--padding_size', default=200, type=int, help='Padding size of sentences.(default=128)')

    params = parser.parse_args()
    print('Parameters:', params.__dict__)
    X_train, X_test, y_train, y_test, vocab = data_loader(params)
    print(X_train)
