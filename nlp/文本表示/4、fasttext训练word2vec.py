import fasttext

path = '/Users/wangrui/Desktop/人工智能教程/阶段5-自然语言处理与NLP/05_深度学习与NLP/01_深度学习与NLPday02/02-代码/data/fil9'
model = fasttext.train_unsupervised(path)
model.save_model('fil9.bin')