import torchvision

dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
print(dataset[0])

img = dataset[0][0]
img.show() #打开图片