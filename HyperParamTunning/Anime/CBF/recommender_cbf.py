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

from collections import defaultdict
from six import iteritems
from scipy.spatial.distance import cosine
from surprise import Reader, Dataset, KNNWithMeans, SVD, NMF
from matplotlib import pyplot as plt
from ..base_recommender import BaseRecommender


with open('../trainset.pickle', 'rb') as f1, open('../validation_set.pickle', 'rb') as f2:
    trainset = pickle.load(f1)
    validation_set = pickle.load(f2)


trainset = list(trainset.to_records(index=False))
validation_set = list(validation_set.to_records(index=False))

users_have_liked_items = set([user for (user, item, rating, taste) in validation_set if taste == 1])

validation_set = [(user, item) for (user, item, rating, taste) in validation_set if taste == 1]
users_for_validate = list(users_have_liked_items)

print(len(users_for_validate))

######
user_items_in_validation_set = defaultdict(list) # dictionary that index each user with relevant items in the validation set
items_in_validation_set = set()
items_in_trainset = set()
user_rated_items = defaultdict(list)
user_unrated_items = defaultdict(list)

for user, item in validation_set:
    user_items_in_validation_set[user].append(item)
    items_in_validation_set.add(item)

for (user, item, rating, taste) in trainset:
    user_rated_items[user].append(item)
    items_in_trainset.add(item)

all_possible_items = set(list(items_in_validation_set) + list(items_in_trainset))

for user in user_items_in_validation_set.keys():
    un_rated_items = [item for item in all_possible_items if item not in user_rated_items[user]]
    user_unrated_items[user] = un_rated_items
######