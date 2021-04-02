import pickle
import json
import numpy as np
import pandas as pd
import collections
import math
import torch
import scipy.sparse as sp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import os

from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from collections import defaultdict
from six import iteritems
from scipy.spatial.distance import cosine
from bayes_opt import BayesianOptimization

from collections import defaultdict
import heapq

######################################################################
##########                                                  ##########
##########                  Load datasets                   ##########
##########                                                  ##########
######################################################################

print("Load datasets...")
work_dir = ''
N_ITERATION_BO = 20

######################################################################
##########                                                  ##########
##########                  Define Optim Functions          ##########
##########                                                  ##########
######################################################################


def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index+2))
    return 0

def metrics(model, test_loader, top_k):
    HR, NDCG = [], []

    for user, item, label in test_loader:
        user = user.cuda()
        item = item.cuda()

        predictions = model(user, item)
        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(
                item, indices).cpu().numpy().tolist()

        gt_item = item[0].item()
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))

    return np.mean(HR), np.mean(NDCG)

print("Datasets loaded")


######################################################################
##########                                                  ##########
##########                  Tune DNN                        ##########
##########                                                  ##########
######################################################################

print("Tune DNN")
train_rating = 'DNN/train.csv'
test_negative = 'DNN/validation.negative'
config_model = 'NeuMF-end'

class NCFData(data.Dataset):
    def __init__(self, features,
                 num_item, train_mat=None, num_ng=0, is_training=None):
        super(NCFData, self).__init__()
        """ Note that the labels are only useful when training, we thus
            add them in the ng_sample() function.
        """
        self.features_ps = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training
        self.labels = [0 for _ in range(len(features))]

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'

        self.features_ng = []
        for x in self.features_ps:
            u = x[0]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_ng.append([u, j])

        labels_ps = [1 for _ in range(len(self.features_ps))]
        labels_ng = [0 for _ in range(len(self.features_ng))]

        self.features_fill = self.features_ps + self.features_ng
        self.labels_fill = labels_ps + labels_ng

    def __len__(self):
        return (self.num_ng + 1) * len(self.labels)

    def __getitem__(self, idx):
        features = self.features_fill if self.is_training \
            else self.features_ps
        labels = self.labels_fill if self.is_training \
            else self.labels

        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]
        return user, item, label


class NCF(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers,
                 dropout, model, GMF_model=None, MLP_model=None):
        super(NCF, self).__init__()

        self.dropout = dropout
        self.model = model
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model

        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.embed_user_MLP = nn.Embedding(
            user_num, factor_num * (2 ** (num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(
            item_num, factor_num * (2 ** (num_layers - 1)))

        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        if self.model in ['MLP', 'GMF']:
            predict_size = factor_num
        else:
            predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weight_()

    def _init_weight_(self):
        """ We leave the weights initialization here. """
        if not self.model == 'NeuMF-pre':
            nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
            nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(self.predict_layer.weight,
                                     a=1, nonlinearity='sigmoid')

            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            # embedding layers
            self.embed_user_GMF.weight.data.copy_(
                self.GMF_model.embed_user_GMF.weight)
            self.embed_item_GMF.weight.data.copy_(
                self.GMF_model.embed_item_GMF.weight)
            self.embed_user_MLP.weight.data.copy_(
                self.MLP_model.embed_user_MLP.weight)
            self.embed_item_MLP.weight.data.copy_(
                self.MLP_model.embed_item_MLP.weight)

            # mlp layers
            for (m1, m2) in zip(
                    self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)

            # predict layers
            predict_weight = torch.cat([
                self.GMF_model.predict_layer.weight,
                self.MLP_model.predict_layer.weight], dim=1)
            precit_bias = self.GMF_model.predict_layer.bias + \
                          self.MLP_model.predict_layer.bias

            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.bias.data.copy_(0.5 * precit_bias)

    def forward(self, user, item):
        if not self.model == 'MLP':
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            output_GMF = embed_user_GMF * embed_item_GMF
        if not self.model == 'GMF':
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
            output_MLP = self.MLP_layers(interaction)

        if self.model == 'GMF':
            concat = output_GMF
        elif self.model == 'MLP':
            concat = output_MLP
        else:
            concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)


def load_all(test_num=100):
    """ We load all the three file here to save time in each epoch. """
    train_data = pd.read_csv(
        train_rating,
        sep='\t', header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1

    train_data = train_data.values.tolist()

    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    test_data = []
    with open(test_negative, 'r') as fd:
      line = fd.readline()
      while line != None and line != '':
        arr = line.strip().split('\t')
        u = eval(arr[0])[0]
        test_data.append([u, eval(arr[0])[1]])
        for i in arr[1:]:
          test_data.append([u, int(i)])
        line = fd.readline()
    return train_data, test_data, user_num, item_num, train_mat


def optim_func_dnn(batch_size, factor_num, num_layers, dropout, lr, epochs, num_ng):
    batch_size = int(batch_size)
    factor_num = int(factor_num)
    num_layers = int(num_layers)
    epochs = int(epochs)
    # print(epochs)
    num_ng = int(num_ng)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    cudnn.benchmark = True
    
    train_data, test_data, user_num, item_num, train_mat = load_all()

    train_dataset = NCFData(train_data, item_num, train_mat, num_ng, True)

    test_dataset = NCFData(test_data, item_num, train_mat, 0, False)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_loader = data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)

    GMF_model, MLP_model = None, None

    nn_model = NCF(user_num, item_num, factor_num, num_layers, dropout, config_model, GMF_model, MLP_model)
    nn_model.cuda()
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(nn_model.parameters(), lr=lr)

    best_ndcg = 0
    for epoch in range(epochs):
        nn_model.train()  # Enable dropout (if have).
        train_loader.dataset.ng_sample()
        # print("Training starts...")
        # iteration = 0
        for user, item, label in train_loader:
            user = user.cuda()
            item = item.cuda()
            label = label.float().cuda()

            nn_model.zero_grad()
            prediction = nn_model(user, item)
            label = label.float()
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()

        nn_model.eval()
        HR, NDCG = metrics(nn_model, test_loader, 10)
        print(NDCG)

        if NDCG > best_ndcg:
            best_ndcg = NDCG

    return best_ndcg


tuning_params_dnn = {
      "batch_size": (64, 256),
      "factor_num": (8, 128),
      "num_layers": (1, 5),
      "dropout": (0, 1),
      "lr": (0.001, 0.01),
      "epochs": (10, 30),
      "num_ng": (0, 10),
}

optimizer_dnn = BayesianOptimization(
    f=optim_func_dnn,
    pbounds=tuning_params_dnn,
    verbose=0, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

logger = JSONLogger(path="DNN/logs_dnn_PO5.json")
optimizer_dnn.subscribe(Events.OPTIMIZATION_STEP, logger)


optimizer_dnn.maximize(
    init_points=0,
    n_iter=N_ITERATION_BO,
)
