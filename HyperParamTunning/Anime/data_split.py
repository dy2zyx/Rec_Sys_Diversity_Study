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
from sklearn.model_selection import train_test_split

path_trainset = '/Users/yudu/Documents/phd/dataset/Anime/trainset_80_1.pickle'
path_testset = '/Users/yudu/Documents/phd/dataset/Anime/testset_20_1.pickle'

with open(path_trainset, 'rb') as trainset, open(path_testset, 'rb') as testset:
    trainset = pickle.load(trainset)
    testset = pickle.load(testset)

print(len(trainset))
print(len(testset))

train, validation_set = train_test_split(trainset, test_size=0.2)
print(len(train))
print(len(validation_set))
print(len(testset))

with open('trainset.pickle', 'wb') as f1, open('validation_set.pickle', 'wb') as f2:
    pickle.dump(train, f1, pickle.HIGHEST_PROTOCOL)
    pickle.dump(validation_set, f2, pickle.HIGHEST_PROTOCOL)
