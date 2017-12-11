import numpy as np
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import svm

from train import gen_seq_data


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

    d = LDA(n_components=30).fit(train_data, train_labels)

    train_data = d.transform(train_data)
    test_data = d.transform(test_data)

    clf = svm.SVC(probability=True).fit(train_data, train_labels)
    pred = clf.predict(test_data)
    print('accuracy: {:0.3f}'.format(np.mean(test_labels == pred)))


if __name__ == '__main__':
    main()
