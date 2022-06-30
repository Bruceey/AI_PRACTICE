import hanlp

content = "工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作"

tokenizer = hanlp.load('CTB6_CONVSEG')

print(tokenizer(content))
