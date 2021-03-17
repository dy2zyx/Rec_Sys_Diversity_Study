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

######################################################################
##########                                                  ##########
##########                  Load Files                      ##########
##########                                                  ##########
######################################################################
path_trainset = '../../Datasets/Movie/trainset.pickle'
path_testset = '../../Datasets/Movie/testset.pickle'

print("Loading datasets...")
with open(path_trainset, 'rb') as trainset, open(path_testset, 'rb') as testset:
    trainset = pickle.load(trainset)
    testset = pickle.load(testset)

sim_dict_file = '/Users/yudu/Documents/phd/dataset/ml1m/diversity_purpose/KG_rec/ExpansionApproach/sim_matrix_ebd.pickle'
with open(sim_dict_file, 'rb') as sim_matrix:
    distance_dict = pickle.load(sim_matrix)

users_have_liked_items = set([user for (user, item, rating) in testset if rating > 3])
test_set = [(user, item) for (user, item, rating) in testset if rating > 3]

users_for_test = list(users_have_liked_items) # the whole set of users in testset

user_relevant_items_in_testset = defaultdict(list)

for user, item in test_set:
    user_relevant_items_in_testset[user].append(item)

######
user_items_in_test_set = defaultdict(list) # dictionary that index each user with relevant items in the testset
items_in_testset = set()
items_in_trainset = set()
user_rated_items = defaultdict(list)
user_unrated_items = defaultdict(list)

for user, item in test_set:
    user_items_in_test_set[user].append(item)
    items_in_testset.add(item)

for (user, item, rating) in trainset:
    user_rated_items[user].append(item)
    items_in_trainset.add(item)

all_possible_items = set(list(items_in_testset) + list(items_in_trainset))

for user in user_items_in_test_set.keys():
    un_rated_items = [item for item in all_possible_items if item not in user_rated_items[user]]
    user_unrated_items[user] = un_rated_items
######

######################################################################
##########                                                  ##########
##########                        CBF                       ##########
##########                                                  ##########
######################################################################
print("Computation starts...")
print("CBF...")

##################################################
##########   Recommender construction   ##########
##################################################
user_positive_profils = defaultdict(list)

for user, item, rating in trainset:
    if rating > 3:
        user_positive_profils[user].append(item)


def predict_rating_cbf(user, item):
    length_profil = len(user_positive_profils[user])
    rating = 0
    if length_profil == 0:
        for i in user_rated_items[user]:
            sim_i_item = 1 - distance_dict[i][item]
            rating += sim_i_item
        return rating / len(user_rated_items[user])
    else:
        for i in user_positive_profils[user]:
            sim_i_item = 1 - distance_dict[i][item]
            rating += sim_i_item
        return rating / length_profil


def recommend_topN_cbf(user):
    top_n = list()
    for un_rated_item in user_unrated_items[user]:
        predicted_rating = predict_rating_cbf(user, un_rated_item)
        top_n.append((un_rated_item, predicted_rating))
    top_n.sort(key=lambda x:x[1], reverse=True)
    return top_n[:1000]


user_candidate_items_dict_cbf = defaultdict(list)
for user in user_unrated_items.keys():
    user_candidate_items = recommend_topN_cbf(user)
    user_candidate_items_dict_cbf[user] = user_candidate_items

with open('candidates_dicts/user_candidate_items_dict_cbf.pickle', 'wb') as f1:
    pickle.dump(user_candidate_items_dict_cbf, f1, pickle.HIGHEST_PROTOCOL)


######################################################################
##########                                                  ##########
##########                        TopPopular                ##########
##########                                                  ##########
######################################################################
print("TopPopular...")

##################################################
##########   Recommender construction   ##########
##################################################

item_popularity_dict = defaultdict(list)
for u, i, r in trainset:
    item_popularity_dict[i].append(u)

popu_max = max([len(item_popularity_dict[i]) for i in item_popularity_dict.keys()])
popu_min = min([len(item_popularity_dict[i]) for i in item_popularity_dict.keys()])

item_popularity_dict_norm = dict()
for item in item_popularity_dict.keys():
    item_popu = (len(item_popularity_dict[item]) - popu_min) / (popu_max - popu_min)
    item_popularity_dict_norm[item] = item_popu


def recommend_topN_top_pop(user):
    top_n = list()
    for un_rated_item in user_unrated_items[user]:
        item_popularity = item_popularity_dict_norm[un_rated_item] if un_rated_item in item_popularity_dict_norm.keys() else 0
        top_n.append((un_rated_item, item_popularity))
    top_n.sort(key=lambda x:x[1], reverse=True)
    return top_n[:1000]


user_candidate_items_dict_top_pop = defaultdict(list)
for user in user_unrated_items.keys():
    user_candidate_items = recommend_topN_top_pop(user)
    user_candidate_items_dict_top_pop[user] = user_candidate_items

with open('candidates_dicts/user_candidate_items_dict_top_pop.pickle', 'wb') as f1:
    pickle.dump(user_candidate_items_dict_top_pop, f1, pickle.HIGHEST_PROTOCOL)

######################################################################
##########                                                  ##########
##########                    CBF-TopPopular                ##########
##########                                                  ##########
######################################################################
print("CBF-TopPopular...")

##################################################
##########   Recommender construction   ##########
##################################################


def recommend_topN_cbf_pop(user, weight):
    top_n = list()
    for un_rated_item in user_unrated_items[user]:
        item_popularity = item_popularity_dict_norm[un_rated_item] if un_rated_item in item_popularity_dict_norm.keys() else 0
        cbf_score = predict_rating_cbf(user, un_rated_item)
        score = weight * item_popularity + (1 - weight) * cbf_score
        top_n.append((un_rated_item, score))
    top_n.sort(key=lambda x:x[1], reverse=True)
    return top_n[:1000]


user_candidate_items_dict_cbf_pop = defaultdict(list)
for user in user_unrated_items.keys():
    user_candidate_items = recommend_topN_cbf_pop(user)
    user_candidate_items_dict_cbf_pop[user] = user_candidate_items

with open('candidates_dicts/user_candidate_items_dict_cbf_pop.pickle', 'wb') as f1:
    pickle.dump(user_candidate_items_dict_cbf_pop, f1, pickle.HIGHEST_PROTOCOL)

######################################################################
##########                                                  ##########
##########                    IBCF                          ##########
##########                                                  ##########
######################################################################
print("IBCF...")

##################################################
##########   Recommender construction   ##########
##################################################
data = pd.DataFrame([(u, i, r) for u, i, r in trainset], columns=['userID', 'itemID', 'rating'])
reader = Reader(rating_scale=(1, 10))
dataset = Dataset.load_from_df(data, reader)
train_set = dataset.build_full_trainset()

ibcf_model = KNNWithZScore(k=40, sim_options={'name': 'cosine', 'user_based': False}, verbose=False)
ibcf_model.fit(train_set)


def recommend_topN_ibcf(user, model):
    top_n = list()
    for un_rated_item in user_unrated_items[user]:
        predicted_rating = model.predict(user, un_rated_item)[3]
        top_n.append((un_rated_item, predicted_rating))
    top_n.sort(key=lambda x: x[1], reverse=True)
    return top_n[:1000]

user_candidate_items_dict_ibcf = defaultdict(list)
for user in user_unrated_items.keys():
    user_candidate_items = recommend_topN_ibcf(user, ibcf_model)
    user_candidate_items_dict_ibcf[user] = user_candidate_items

with open('candidates_dicts/user_candidate_items_dict_ibcf.pickle', 'wb') as f1:
    pickle.dump(user_candidate_items_dict_ibcf, f1, pickle.HIGHEST_PROTOCOL)

######################################################################
##########                                                  ##########
##########                    SVD                           ##########
##########                                                  ##########
######################################################################
print("SVD...")

##################################################
##########   Recommender construction   ##########
##################################################
svd_model = SVD()
svd_model.fit(train_set)


def recommend_topN_svd(user, model):
    top_n = list()
    for un_rated_item in user_unrated_items[user]:
        predicted_rating = model.predict(user, un_rated_item)[3]
        top_n.append((un_rated_item, predicted_rating))
    top_n.sort(key=lambda x: x[1], reverse=True)
    return top_n[:1000]

user_candidate_items_dict_svd = defaultdict(list)
for user in user_unrated_items.keys():
    user_candidate_items = recommend_topN_svd(user, svd_model)
    user_candidate_items_dict_svd[user] = user_candidate_items

with open('candidates_dicts/user_candidate_items_dict_svd.pickle', 'wb') as f1:
    pickle.dump(user_candidate_items_dict_svd, f1, pickle.HIGHEST_PROTOCOL)

######################################################################
##########                                                  ##########
##########                    DNN                           ##########
##########                                                  ##########
######################################################################
print("DNN...")
import sys
sys.path.insert(0, '/Users/yudu/PycharmProjects/NCF')

##################################################
##########   Recommender construction   ##########
##################################################
model_file = "model.pth"
neuMF = torch.load(model_file)
neuMF.eval()


def recommend_topN_dnn(user, model):
    top_n = list()
    for un_rated_item in user_unrated_items[user]:
        predicted_rating = model(torch.tensor([[user]]), torch.tensor([[un_rated_item]]))
        predicted_rating = predicted_rating.item()
        top_n.append((un_rated_item, predicted_rating))
    top_n.sort(key=lambda x:x[1], reverse=True)
    return top_n[:1000]

user_candidate_items_dict_dnn = defaultdict(list)
for user in user_unrated_items.keys():
    user_candidate_items = recommend_topN_dnn(user, neuMF)
    user_candidate_items_dict_dnn[user] = user_candidate_items

with open('candidates_dicts/user_candidate_items_dict_dnn.pickle', 'wb') as f1:
    pickle.dump(user_candidate_items_dict_dnn, f1, pickle.HIGHEST_PROTOCOL)

######################################################################
##########                                                  ##########
##########                      KGE                         ##########
##########                                                  ##########
######################################################################
print("KGE...")
trainset_entity_embedding_file = 'dict_entity_embedding.pickle'
trainset_relation_embedding_file = 'dict_relation_embedding.pickle'

BASE_URI = "http://example.org/rating_ontology"
USER = BASE_URI + "/User/User_"
ITEM = BASE_URI + "/Item/Item_"
TASTE = BASE_URI + "/Taste#"

with open(trainset_entity_embedding_file, 'rb') as f1, open(trainset_relation_embedding_file, 'rb') as f2:
    dict_entity_embedding = pickle.load(f1)
    dict_relation_embedding = pickle.load(f2)


def recommend_topN_kge(user):
    top_n = list()
    for un_rated_item in user_unrated_items[user]:
        user_embedding = dict_entity_embedding[USER + str(user)]
        r_like_embedding = dict_relation_embedding[TASTE + 'like']
        item_embedding = dict_entity_embedding[ITEM + str(un_rated_item)]
        score = compute_score(user_embedding, r_like_embedding, item_embedding)
        top_n.append((un_rated_item, score))
    top_n.sort(key=lambda x:x[1], reverse=True)
    return top_n[:1000]


def compute_score(h_embedding, r_embedding, t_embedding):
    sum_vec = np.asarray(h_embedding) + np.asarray(r_embedding) - np.asarray(t_embedding)
    distance_l1 = np.linalg.norm(sum_vec, ord=1)
    return -distance_l1


user_candidate_items_dict_kge = defaultdict(list)
for user in user_unrated_items.keys():
    user_candidate_items = recommend_topN_kge(user)
    user_candidate_items_dict_kge[user] = user_candidate_items

with open('candidates_dicts/user_candidate_items_dict_kge.pickle', 'wb') as f1:
    pickle.dump(user_candidate_items_dict_kge, f1, pickle.HIGHEST_PROTOCOL)