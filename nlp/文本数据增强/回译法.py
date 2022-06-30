"""
踩坑链接
https://blog.csdn.net/weixin_39574469/article/details/117886420
"""
from googletrans import Translator, LANGUAGES
import json

p_sample1 = "酒店设施非常不错"
p_sample2 = "这家价格很便宜"
n_sample1 = "拖鞋都发霉了，太差了"
n_sample2 = "电视不好用，没有看到足球"

translator = Translator(service_urls=[
      'translate.google.cn',
    ])
text = json.dumps([p_sample1, p_sample2, n_sample1, n_sample2], ensure_ascii=False)
translations = translator.translate(text, dest='ko')
ko_res = translations.text

print("翻译为韩文的结果为：", ko_res, sep='\n')

#回译
translations = translator.translate(json.dumps(ko_res), dest='zh-cn')
zh_res = translations.text
print("回译为中文的结果为：", zh_res, sep='\n')
print("最原始的评论为：", [p_sample1, p_sample2, n_sample1, n_sample2], sep='\n')