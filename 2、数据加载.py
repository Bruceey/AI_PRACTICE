from torch.utils.data import Dataset, DataLoader
import pandas as pd



class CifarDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path) as f:
            lines = [[i[:4].strip(), i[4:].strip()] for i in f]
        self.df = pd.DataFrame(lines, columns=['label', 'sms'])

    def __getitem__(self, index):
        single_item = self.df.iloc[index, :]
        return single_item.values[0], single_item.values[1]

    def __len__(self):
        return self.df.shape[0]


data_path = "data/smsspamcollection/SMSSpamCollection"

d = CifarDataset(data_path)
# for i in range(len(d)):
#     print(i, d[i])

data_loader = DataLoader(dataset=d, batch_size=10, shuffle=True)

for index, (label, context) in enumerate(data_loader):
    print(index, label, context)
    print('*' * 100)