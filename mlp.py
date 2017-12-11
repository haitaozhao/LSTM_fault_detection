from datetime import datetime
import numpy as np
import torch
import torch.utils.data as tchdata
from sklearn import preprocessing

from train import AccMectric, gen_seq_data, train, validate

class TEMlp(torch.nn.Module):
    def __init__(self, i, h, o, ):
        super(TEMlp, self).__init__()
        self.h1 = torch.nn.Linear(i, h)
        self.b1 = torch.nn.BatchNorm1d(h)
        self.a1 = torch.nn.LeakyReLU(0.01, True)
        self.h2 = torch.nn.Linear(h, h)
        self.b2 = torch.nn.BatchNorm1d(h)
        self.a2 = torch.nn.LeakyReLU(0.01, True)
        self.sm = torch.nn.Linear(h, o)
        
    def forward(self, x):
        x = self.a1(self.b1(self.h1(x)))
        x = self.a2(self.b2(self.h2(x)))
        x = self.sm(x)
        return x


def main():
    n_samples = 3                           # 1, 3 or 5
    n_hidden = 30

    # target = [1, 2, 6, 7, 8]              # case1
    # target = [3, 4, 5, 9, 10, 11, 12]     # case2
    target = list(range(1, 22))           # total

    train_data, train_labels = gen_seq_data(target, n_samples, is_train=True)
    test_data, test_labels = gen_seq_data(target, n_samples, is_train=False)

    scaler = preprocessing.StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    train_dataset = tchdata.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels))
    test_dataset = tchdata.TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels))

    train_loader = tchdata.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = tchdata.DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = TEMlp(52 * n_samples, n_hidden, len(target))
    model.cuda()
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.005)

    for i in range(60):
        train_acc = train(model, optimizer, train_loader)
        test_acc = validate(model, test_loader)
        print('{}\tepoch = {}\ttrain accuracy: {:0.3f}\ttest accuracy: {:0.3f}' \
            .format(datetime.now(), i, train_acc, test_acc))


if __name__ == '__main__':
    main()
