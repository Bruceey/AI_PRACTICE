import jieba.posseg as pseg
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from itertools import chain

def get_adj_str(text):
    """
    用于获取adj的字符串，并用空格隔开
    :param text:
    :return:
    """
    r = ''
    for g in pseg.cut(text):      # g的类型pair(词，词性)，
        if g.flag == 'a':
            r += g.word
    return r


def get_word_cloud(keywords: str):
    font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'
    word_cloud = WordCloud(font_path=font_path, max_words=100, background_color='white')
    word_cloud.generate(keywords)

    #绘图
    plt.figure()
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

train_data = pd.read_csv('/Users/wangrui/Desktop/人工智能教程/阶段5-自然语言处理与NLP/05_深度学习与NLP/01_深度学习与NLPday02/02-代码/text_preprocess/cn_data/train.tsv', sep='\t')
valid_data = pd.read_csv('/Users/wangrui/Desktop/人工智能教程/阶段5-自然语言处理与NLP/05_深度学习与NLP/01_深度学习与NLPday02/02-代码/text_preprocess/cn_data/dev.tsv', sep='\t')

#获得训练集上正样本
p_train_data = train_data[train_data.label == 1]['sentence']

train_p_a_vocab = ' '.join(map(lambda x: get_adj_str(x), p_train_data))

#负训练集
n_train_data = train_data[train_data.label == 0]['sentence']
train_n_a_vocab = ' '.join(map(lambda x: get_adj_str(x), n_train_data))

get_word_cloud(train_p_a_vocab)
# get_word_cloud(train_n_a_vocab)
