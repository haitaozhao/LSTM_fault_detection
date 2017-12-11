from datetime import datetime
import numpy as np
import torch
import torch.utils.data as tchdata
from sklearn import preprocessing

from train import AccMectric, gen_seq_data, train, validate


class TELSTM(torch.nn.Module):
    def __init__(self, i, h, o, n_samples, is_bn=False):
        super(TELSTM, self).__init__()
        self._lstm_cell = torch.nn.LSTMCell(i, h)
        self._fc = torch.nn.Linear(h, o)
        self._hidden = h
        self._n_samples = n_samples
        self._is_bn = is_bn
        if self._is_bn:
            self._bn = torch.nn.BatchNorm1d(h)
        
    def forward(self, x):
        seq_data = x.chunk(self._n_samples, dim=1)
        h_t = torch.autograd.Variable(torch.zeros(x.size(0), self._hidden).cuda())
        c_t = torch.autograd.Variable(torch.zeros(x.size(0), self._hidden).cuda())
        for data in seq_data:
            h_t, c_t = self._lstm_cell(data, (h_t, c_t))
        if self._is_bn:
            h_t = self._bn(h_t)
        fc = self._fc(h_t)
        return fc


def main():
    n_samples = 3                           # 3 or 5
    n_hidden = 30

    target = [1, 2, 6, 7, 8]              # case1
    # target = [3, 4, 5, 9, 10, 11, 12]     # case2
    # target = list(range(1, 22))           # total

    train_data, train_labels = gen_seq_data(target, n_samples, is_train=True)
    test_data, test_labels = gen_seq_data(target, n_samples, is_train=False)

    scaler = preprocessing.StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    train_dataset = tchdata.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels))
    test_dataset = tchdata.TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels))

    train_loader = tchdata.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = tchdata.DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = TELSTM(52, n_hidden, len(target), n_samples, False)
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
