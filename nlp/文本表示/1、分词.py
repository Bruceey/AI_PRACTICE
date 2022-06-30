import jieba
jieba.load_userdict("./word")

content = "工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作"

res = jieba.lcut(content)

# res = jieba.lcut_for_search(content)
# print(res)


