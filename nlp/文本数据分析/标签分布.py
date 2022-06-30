import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("fivethirtyeight")

train_data = pd.read_csv('/Users/wangrui/Desktop/人工智能教程/阶段5-自然语言处理与NLP/05_深度学习与NLP/01_深度学习与NLPday02/02-代码/text_preprocess/cn_data/train.tsv', sep='\t')
valid_data = pd.read_csv('/Users/wangrui/Desktop/人工智能教程/阶段5-自然语言处理与NLP/05_深度学习与NLP/01_深度学习与NLPday02/02-代码/text_preprocess/cn_data/dev.tsv', sep='\t')

sns.countplot('label', data=train_data)
plt.title('train_data')
plt.show()

sns.countplot('label', data=valid_data)
plt.title('valid_data')
plt.show()