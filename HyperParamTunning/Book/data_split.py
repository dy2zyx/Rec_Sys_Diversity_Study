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

path_trainset = '/Users/yudu/Documents/phd/dataset/lbthing/rating_dataset/trainset.dat'
path_testset = '/Users/yudu/Documents/phd/dataset/lbthing/rating_dataset/testset.dat'

trainset = pd.read_csv(path_trainset, names=['uid', 'iid', 'rating'], sep=' ')
testset = pd.read_csv(path_testset, names=['uid', 'iid', 'rating'], sep=' ')

trainset = list(trainset.to_records(index=False))
testset = list(testset.to_records(index=False))

print(len(trainset))
print(len(testset))

trainset, validation_set = train_test_split(trainset, test_size=0.2)
print(len(trainset))
print(len(validation_set))
print(len(testset))

with open('trainset.pickle', 'wb') as f1, open('validation_set.pickle', 'wb') as f2, open('testset.pickle', 'wb') as f3:
    pickle.dump(trainset, f1, pickle.HIGHEST_PROTOCOL)
    pickle.dump(validation_set, f2, pickle.HIGHEST_PROTOCOL)
    pickle.dump(testset, f3, pickle.HIGHEST_PROTOCOL)