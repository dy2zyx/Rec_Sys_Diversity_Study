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
path_trainset = '../../Datasets/Anime/trainset.pickle'
path_testset = '../../Datasets/Anime/testset.pickle'

print("Loading datasets...")
with open(path_trainset, 'rb') as trainset, open(path_testset, 'rb') as testset:
    trainset = pickle.load(trainset)
    testset = pickle.load(testset)

sim_dict_file = 'sim_matrix_ebd.pickle'
with open(sim_dict_file, 'rb') as sim_matrix:
    distance_dict = pickle.load(sim_matrix)

users_have_liked_items = set([user for (user, item, rating) in testset if rating > 7])
test_set = [(user, item) for (user, item, rating) in testset if rating > 7]

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

#####################################################################
#########                                                  ##########
#########                        CBF                       ##########
#########                                                  ##########
#####################################################################
print("Computation starts...")
print("CBF...")

##################################################
##########   Recommender construction   ##########
##################################################
user_positive_profils = defaultdict(list)

for user, item, rating in trainset:
    if rating > 7:
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
    return top_n[:500]


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


# def recommend_topN_top_pop(user):
#     top_n = list()
#     for un_rated_item in user_unrated_items[user]:
#         item_popularity = item_popularity_dict_norm[un_rated_item] if un_rated_item in item_popularity_dict_norm.keys() else 0
#         top_n.append((un_rated_item, item_popularity))
#     top_n.sort(key=lambda x:x[1], reverse=True)
#     return top_n[:500]


# user_candidate_items_dict_top_pop = defaultdict(list)
# for user in user_unrated_items.keys():
#     user_candidate_items = recommend_topN_top_pop(user)
#     user_candidate_items_dict_top_pop[user] = user_candidate_items

# with open('candidates_dicts/user_candidate_items_dict_top_pop.pickle', 'wb') as f1:
#     pickle.dump(user_candidate_items_dict_top_pop, f1, pickle.HIGHEST_PROTOCOL)


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
    return top_n[:500]

__WEIGHT = 0.4176 # best value

user_candidate_items_dict_cbf_pop = defaultdict(list)
for user in user_unrated_items.keys():
    user_candidate_items = recommend_topN_cbf_pop(user, __WEIGHT)
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

K__ = 1

ibcf_model = KNNWithZScore(k=K__, sim_options={'name': 'cosine', 'user_based': False}, verbose=False)
ibcf_model.fit(train_set)


def recommend_topN_ibcf(user, model):
    top_n = list()
    for un_rated_item in user_unrated_items[user]:
        predicted_rating = model.predict(user, un_rated_item)[3]
        top_n.append((un_rated_item, predicted_rating))
    top_n.sort(key=lambda x: x[1], reverse=True)
    return top_n[:500]

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
lr_all, n_epochs, n_factors, reg_all = 0.0003, 3, 200, 0.0626
svd_model = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
svd_model.fit(train_set)


def recommend_topN_svd(user, model):
    top_n = list()
    for un_rated_item in user_unrated_items[user]:
        predicted_rating = model.predict(user, un_rated_item)[3]
        top_n.append((un_rated_item, predicted_rating))
    top_n.sort(key=lambda x: x[1], reverse=True)
    return top_n[:500]

user_candidate_items_dict_svd = defaultdict(list)
for user in user_unrated_items.keys():
    user_candidate_items = recommend_topN_svd(user, svd_model)
    user_candidate_items_dict_svd[user] = user_candidate_items

with open('candidates_dicts/user_candidate_items_dict_svd.pickle', 'wb') as f1:
    pickle.dump(user_candidate_items_dict_svd, f1, pickle.HIGHEST_PROTOCOL)

#####################################################################
#########                                                  ##########
#########                    DNN                           ##########
#########                                                  ##########
#####################################################################
print("DNN...")
# import sys
# sys.path.insert(0, '/Users/yudu/PycharmProjects/NCF')

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

##################################################
##########   Recommender construction   ##########
##################################################
model_file = "../../HyperParamTunning/Anime/DNN/NeuMF-end_anime.pth"
device = torch.device('cpu')
neuMF = torch.load(model_file, map_location=device)
neuMF.eval()


def recommend_topN_dnn(user, model):
    top_n = list()
    for un_rated_item in user_unrated_items[user]:
        predicted_rating = model(torch.tensor([[user]]), torch.tensor([[un_rated_item]]))
        predicted_rating = predicted_rating.item()
        top_n.append((un_rated_item, predicted_rating))
    top_n.sort(key=lambda x:x[1], reverse=True)
    return top_n[:500]

user_candidate_items_dict_dnn = defaultdict(list)
for user in user_unrated_items.keys():
    user_candidate_items = recommend_topN_dnn(user, neuMF)
    user_candidate_items_dict_dnn[user] = user_candidate_items

with open('candidates_dicts/user_candidate_items_dict_dnn.pickle', 'wb') as f1:
    pickle.dump(user_candidate_items_dict_dnn, f1, pickle.HIGHEST_PROTOCOL)

#####################################################################
#########                                                  ##########
#########                      KGE                         ##########
#########                                                  ##########
#####################################################################
print("KGE...")
trainset_entity_embedding_file = '../../HyperParamTunning/Anime/KGE/train_model/dict_entity_embedding.pickle'
trainset_relation_embedding_file = '../../HyperParamTunning/Anime/KGE/train_model/dict_relation_embedding.pickle'

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
    return top_n[:500]


def compute_score(h_embedding, r_embedding, t_embedding):
    sum_vec = np.asarray(h_embedding) + np.asarray(r_embedding) - np.asarray(t_embedding)
    distance_l1 = np.linalg.norm(sum_vec, ord=1)
    return -distance_l1


user_candidate_items_dict_kge = defaultdict(list)
for user in user_unrated_items.keys():
    if USER + str(user) in dict_entity_embedding.keys():
        user_candidate_items = recommend_topN_kge(user)
        user_candidate_items_dict_kge[user] = user_candidate_items

with open('candidates_dicts/user_candidate_items_dict_kge.pickle', 'wb') as f1:
    pickle.dump(user_candidate_items_dict_kge, f1, pickle.HIGHEST_PROTOCOL)