import copy
import csv
import time

import numpy as np
from matplotlib import pyplot as plt

from sklearn import naive_bayes
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import roc_auc_score

import torch
from torch import nn

import tqdm
import torch.nn.functional as F

import graphing


# from torchvision.transforms import ToTensor


def run_random_forest(X_train, y_train, X_test, y_test, quick_train=True):
    print('Running Random Forest')
    init_time = time.time()
    if quick_train:
        # ~5 mins per run. Still extremely high roc auc
        clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=1, max_features=10, verbose=3)
    else:
        # ~20 mins per run, slightly more accurate:
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=7, max_features=30, verbose=3)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # val_pred = clf.predict(X_val)
    y_score = clf.predict_proba(X_test)
    print('y_score:', y_score)

    accuracy = float(np.sum(y_test == y_pred)) / len(y_test)
    print('Random Forest Accuracy: ', accuracy)
    # val_accuracy = float(np.sum(y_val == val_pred)) / len(y_val)
    # print('Random Forest Validation Accuracy: ', val_accuracy)
    print('Elapsed Time: ', time.time()-init_time)
    roc_score = roc_auc_score(y_test, y_score, multi_class='ovr', average='micro')
    print("Area under roc curve percentage:", roc_score, "%")
    print(y_score.shape)
    return clf, y_pred, y_score, clf.classes_


def run_neural_network(X_param, y_param):
    # Help from: https://machinelearningmastery.com/building-a-multiclass-classification-model-in-pytorch/
    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Encoding classes
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y_param)
    # print('Categories:', ohe.categories_)
    y_encode = ohe.transform(y_param)

    # y_encode = y_param
    print(y_encode)
    # print(X_param.shape)

    # convert pandas DataFrame (X) and numpy array (y) into PyTorch tensors
    X = torch.tensor(X_param.to_numpy(), dtype=torch.float32)
    y = torch.tensor(y_encode, dtype=torch.float32)
    y_pred = None
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

    learning_rate = 0.001  # Common starting point for learning rate

    # Define model
    class Multiclass(nn.Module):
        def __init__(self):
            super(Multiclass, self).__init__()
            # self.flatten = nn.Flatten()
            self.hidden = nn.Linear(54, 128)
            self.act = nn.ReLU()
            self.layer = nn.Linear(128, 128)
            self.dropout = nn.Dropout(0.3)
            self.hidden_layer = nn.Linear(128, 128)
            self.output = nn.Linear(128, 9)

        def forward(self, x):
            x = self.act(self.hidden(x))
            x = self.layer(x)
            x = self.dropout(x)
            x = self.hidden_layer(x)
            # x = self.dropout(x)
            x = self.output(x)
            return x

    model = Multiclass()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # prepare model and training parameters
    n_epochs = 200
    batch_size = 10
    batches_per_epoch = len(X_train) // batch_size

    best_acc = - np.inf  # init to negative infinity
    best_y_score = None
    best_weights = None
    train_loss_hist = []
    train_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []
    start_time = time.time()
    # training loop
    for epoch in range(n_epochs):
        epoch_loss = []
        epoch_acc = []
        # set model in training mode and run through each batch
        model.train()
        with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0, colour='#701fc2') as bar:
            bar.set_description(f"Epoch {epoch}")
            for i in bar:
                # take a batch
                start = i * batch_size
                X_batch = X_train[start:start + batch_size]
                y_batch = y_train[start:start + batch_size]
                # forward pass
                y_pred = model(X_batch)

                loss = loss_func(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # compute and store metrics
                acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
                epoch_loss.append(float(loss))
                epoch_acc.append(float(acc))
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
        # set model in evaluation mode and run through the test set
        model.eval()
        y_pred = model(X_test)
        probabilites = F.softmax(y_pred, dim=1)[:, 1]
        y_score = probabilites.detach().numpy()

        ce = loss_func(y_pred, y_test)
        acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean()
        # print(f"Predictions: {torch.argmax(y_pred, 1)}")
        # print(f"Targets: {y_batch}")

        ce = float(ce)
        acc = float(acc)
        train_loss_hist.append(np.mean(epoch_loss))
        train_acc_hist.append(np.mean(epoch_acc))
        test_loss_hist.append(ce)
        test_acc_hist.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_y_score = y_score
            best_weights = copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch} validation: Cross-entropy={ce:.2f}, Accuracy={acc * 100:.1f}%")

    # Restore best model
    model.load_state_dict(best_weights)
    print(f'Total time: {(time.time() - start_time) / 60} minutes')
    graphing.plot_loss(test_loss_hist, train_loss_hist)
    graphing.plot_acc(test_acc_hist, train_acc_hist)
    return model, ohe.inverse_transform(y_pred.detach()), y_test, y_train, y_score


def randomize_random_forest(X, y):
    print('randomizing random forest')
    clf = RandomForestClassifier()
    param_dist = {
        'n_estimators': [300, 400],
        'min_samples_leaf': [1, 3, 7, 20, 50],
        'max_features': ['sqrt', 'log2', None, 5, 10, 30]
    }
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search, verbose=3, scoring='roc_auc_ovr')
    start = time.time()
    print('starting model fit')
    print('starting time: ', start)
    random_search.fit(X, y)
    print(
        "RandomizedSearchCV took %.2f minutes for %d candidates parameter settings."
        % ((time.time() - start)/60, n_iter_search)
    )
    report(random_search.cv_results_)


# Utility function to report best scores
def report(results, n_top=20):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")
    print(results)
    print(type(results))
    with open('results.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(results)


def run_naive_bayes(X_train, y_train, X_test, y_test):
    print('Running Naive Bayes')
    gnb = naive_bayes.GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    accuracy = float(np.sum(y_test == y_pred)) / len(y_test)
    print('Naive Bayes Accuracy: ', accuracy)
    return y_pred


def run_logistic_regression(X_train, y_train, X_test, y_test):
    print('Running Logistic Regression')
    model = LogisticRegression()
    # y_score = model.fit(X_train, y_train).predict_proba(X_test)
    y_score = 0

    y_pred = model.predict(X_test)

    # label_binarizer = LabelBinarizer().fit(y_train)
    # y_onehot_test = label_binarizer.transform(y_test)

    # print(y_pred)
    # print(y_test.tolist())
    # print(type(y_test))
    y_test = y_test.tolist()
    accuracy = float(np.sum(y_test == y_pred)) / len(y_test)
    print('Logistic Regression Accuracy: ', accuracy)

    return y_pred, y_score
