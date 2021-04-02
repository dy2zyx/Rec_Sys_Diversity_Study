import pickle
import json
import numpy as np
import pandas as pd
import collections
import math
import torch


from collections import defaultdict
from six import iteritems
from scipy.spatial.distance import cosine
from surprise import Reader, Dataset, KNNWithZScore, SVD, NMF
from matplotlib import pyplot as plt

path_trainset = '../../../Datasets/Movie/trainset.pickle'

print("Loading datasets...")
with open(path_trainset, 'rb') as trainset:
    trainset = pickle.load(trainset)

print("SVD...")
data = pd.DataFrame([(u, i, r) for u, i, r in trainset], columns=['userID', 'itemID', 'rating'])
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(data, reader)
train_set = dataset.build_full_trainset()

lr_all, n_epochs, n_factors, reg_all = 0.0354, 11, 175, 0.0813

svd_model = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
svd_model.fit(train_set)

with open('trained_svd.pickle', 'wb') as f:
    pickle.dump(svd_model, f, pickle.HIGHEST_PROTOCOL)