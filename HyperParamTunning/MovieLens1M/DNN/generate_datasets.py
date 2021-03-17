import random
import pickle
import json
import time
import numpy as np
import pandas as pd
import collections
import seaborn as sns
import operator
import math

from SPARQLWrapper import SPARQLWrapper2, SPARQLWrapper
from SPARQLWrapper import XML, RDFXML
from collections import defaultdict
from six import iteritems
from scipy.spatial.distance import cosine
from surprise import Reader, Dataset, KNNWithMeans, SVD, NMF
from matplotlib import pyplot as plt






if __name__=='__main__':

    path_trainset = '../trainset.pickle'
    path_validation_set = '../validation_set.pickle'

    with open(path_trainset, 'rb') as train, open(path_validation_set, 'rb') as validation:
        trainset = pickle.load(train)
        validation_set = pickle.load(validation)

    with open('train.csv', mode='w') as output:
        for u, i, r in trainset:
            output.write(str(u) + "\t" + str(i) + "\t" + str(r) + "\n")

    user_ratings_in_validation = defaultdict(list)
    for u, i, r in validation_set:
        user_ratings_in_validation[u].append((u, i, r))

    with open('validation.csv', mode='w') as output:
        for user in user_ratings_in_validation.keys():
            u_ratings = user_ratings_in_validation[user]
            u_pos_ratings = [(u, i, r) for u, i, r in u_ratings if r > 3]
            if not len(u_pos_ratings) == 0:
                for u, i, r in u_pos_ratings:
                    output.write(str(u) + "\t" + str(i) + "\t" + str(r) + "\n")

    # items_in_train = set()
    # user_ratings_in_train = defaultdict(list)
    # user_unrated_items = dict()
    #
    # for u, i, r in trainset:
    #     user_ratings_in_train[u].append((u, i, r))
    #     items_in_train.add(i)
    #
    # for user in user_ratings_in_train.keys():
    #     rated_items = [item for u, item, r in user_ratings_in_train[user]]
    #     unrated_items = [item for item in items_in_train if item not in rated_items]
    #     user_unrated_items[user] = unrated_items
    # # print(user_unrated_items.keys())
    # with open("validation.negative", 'w') as f1, open('validation.csv', 'r') as f2:
    #     samples = f2.readlines()
    #     for sample in samples:
    #         user = sample.split("\t")[0]
    #         item = sample.split("\t")[1]
    #         rating = sample.split("\t")[2]
    #         negative_samples = random.choices(user_unrated_items[int(user)], k=99)
    #         #negative_samples = user_unrated_items[user]
    #         f1.write('(' + str(user) + ',' + str(item) + ')' + "\t")
    #         for negative_sample in negative_samples:
    #             f1.write(str(negative_sample))
    #             f1.write("\t")
    #         f1.write("\n")
