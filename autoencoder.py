from datetime import datetime
import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as tchdata
from sklearn import preprocessing

from train import gen_seq_data

class TEMlp(torch.nn.Module):
    def __init__(self, i, h, o):
        super(TEMlp, self).__init__()
        self.h = torch.nn.Linear(i, h)
        self.a = torch.nn.Tanh()
        self.o = torch.nn.Linear(h, o)
        
    def forward(self, x):
        x = self.o(self.a(self.h(x)))
        return x

    def predict(self, data_loader):
        pred = []
        for data, _ in data_loader:
            x = torch.autograd.Variable(data.cuda())
            o = self.h(x)
            pred.append(o.data.cpu().numpy())
        return np.vstack(pred)


class Metirc(object):

    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, pred, target):
        self._sum += np.sum(np.mean(np.power(target - pred, 2), axis=1))
        self._count += pred.shape[0]

    def get(self):
        return self._sum / self._count


def train(model, optimizer, train_loader):
    model.train()
    l2norm = Metirc()
    for data, _ in train_loader:
        x = torch.autograd.Variable(data.cuda())
        y = model(x)

        loss = torch.mean(torch.pow(x - y, 2))
        l2norm.update(x.data.cpu().numpy(), y.data.cpu().numpy())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return l2norm.get()


def validate(model, val_loader):
    model.eval()
    l2norm = Metirc()
    for data, _ in val_loader:
        x = torch.autograd.Variable(data.cuda())
        y = model(x)
        l2norm.update(y.data.cpu().numpy(), x.data.cpu().numpy())
    return l2norm.get()


def main():
    n_samples = 1                           # 1, 3 or 5
    # n_hidden = 30

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

#     model = TEMlp(52 * n_samples, n_hidden, 52 * n_samples)
#     model.cuda()
#     torch.backends.cudnn.benchmark = True
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.005)
# 
#     for i in range(500):
#         train_acc = train(model, optimizer, train_loader)
#         test_acc = validate(model, test_loader)
#         print('{}\tepoch = {}\ttrain accuracy: {:0.3f}\ttest accuracy: {:0.3f}' \
#             .format(datetime.now(), i, train_acc, test_acc))
# 
#     pred = model.predict(test_loader)
#     save()

    result = {}
    for h in range(2, 32, 2):
        model = TEMlp(52 * n_samples, h, 52 * n_samples)
        model.cuda()
        torch.backends.cudnn.benchmark = True
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.005)
    
        for i in range(500):
            train_acc = train(model, optimizer, train_loader)
            test_acc = validate(model, test_loader)
            print('{}\tepoch = {}\ttrain accuracy: {:0.3f}\ttest accuracy: {:0.3f}' \
                .format(datetime.now(), i, train_acc, test_acc))
    
        pred = model.predict(test_loader)
        result['autoencoder_case1_dim{}'.format(h)] = pred
    sio.savemat('autoencoder.mat', result)


if __name__ == '__main__':
    main()
