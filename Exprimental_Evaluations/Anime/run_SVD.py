#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/11/2020
@author: Yu DU
"""


### Packages ###
import pickle
import json
import numpy as np
import pandas as pd
import collections
import math
import torch
import torch.utils.data as data

import sys
sys.path.insert(0, '../Recommenders/DNN')

from collections import defaultdict
from six import iteritems
from scipy.spatial.distance import cosine


### Global variables settings ###

path_trainset = '../../Datasets/Anime/trainset.pickle'
path_testset = '../../Datasets/Anime/testset.pickle'
items_embedding_file = '../../Datasets/Anime/embeddings/ICKG/itemURI_embedding.json'
mapping_file = '../../Datasets/Anime/mappings/Mapping_Anime-Dbpedia.csv'
trainset_entity_embedding_file = '../../Datasets/Anime/embeddings/ICKG+UPKG/dict_entity_embedding.pickle'
trainset_relation_embedding_file = '../../Datasets/Anime/embeddings/ICKG+UPKG/dict_relation_embedding.pickle'

BASE_URI = "http://example.org/rating_ontology"
USER = BASE_URI + "/User/User_"
ITEM = BASE_URI + "/Item/Item_"
TASTE = BASE_URI + "/Taste#"

with open(trainset_entity_embedding_file, 'rb') as f1, open(trainset_relation_embedding_file, 'rb') as f2:
    dict_entity_embedding = pickle.load(f1)
    dict_relation_embedding = pickle.load(f2)

user_profil_ild_file = '../../Datasets/Anime/data_dicts/user_profile_ILD_dict.pickle'
with open(user_profil_ild_file, 'rb') as user_ilds:
    user_profil_ild_dict = pickle.load(user_ilds)

sim_dict_file = '../../Datasets/Anime/data_dicts/item_semantic_distance_dict.pickle'
with open(sim_dict_file, 'rb') as sim_matrix:
    distance_dict = pickle.load(sim_matrix)

### trainset and testset loading ###

with open(path_trainset, 'rb') as trainset, open(path_testset, 'rb') as testset:
    trainset = pickle.load(trainset)
    testset = pickle.load(testset)

trainset = list(trainset.to_records(index=False))
testset = list(testset.to_records(index=False))

users_have_liked_items = set([user for (user, item, rating, taste) in testset if taste == 1 and USER+str(user) in dict_entity_embedding.keys()])
test_set = [(user, item) for (user, item, rating, taste) in testset if taste == 1 and USER+str(user) in dict_entity_embedding.keys()]

users_for_test = list(users_have_liked_items) # the whole set of users in testset

user_relevant_items_in_testset = defaultdict(list)

for user, item in test_set:
    user_relevant_items_in_testset[user].append(item)

user_items_in_test_set = defaultdict(list) # dictionary that index each user with relevant items in the testset
items_in_testset = set()
items_in_trainset = set()
user_rated_items = defaultdict(list)
user_unrated_items = defaultdict(list)

for user, item in test_set:
    user_items_in_test_set[user].append(item)
    items_in_testset.add(item)

for (user, item, rating, taste) in trainset:
    user_rated_items[user].append(item)
    items_in_trainset.add(item)

all_possible_items = set(list(items_in_testset) + list(items_in_trainset))

for user in user_items_in_test_set.keys():
    un_rated_items = [item for item in all_possible_items if item not in user_rated_items[user]]
    user_unrated_items[user] = un_rated_items

### recommender construction and rating predictions ###

data = pd.DataFrame([(u, i, r) for u, i, r, t in trainset], columns=['userID', 'itemID', 'rating'])
reader = Reader(rating_scale=(1, 10))
dataset = Dataset.load_from_df(data, reader)

train_set = dataset.build_full_trainset()

svd = SVD()
svd.fit(train_set)


"""function to recommend the topN items for a user given the rating prediction model"""
def recommend_topN(user, model):
    top_n = list()
    for un_rated_item in user_unrated_items[user]:
        predicted_rating = model.predict(user, un_rated_item)[3]
        normalised_r = np.divide(predicted_rating-1,9)
        top_n.append((un_rated_item, normalised_r))
    top_n.sort(key=lambda x:x[1], reverse=True)
    return top_n


user_candidate_items_dict = defaultdict(list)
for user in user_unrated_items.keys():
    user_candidate_items = recommend_topN(user, svd)
    user_candidate_items_dict[user] = user_candidate_items


### Evaluation Metrics ###

### Diversity metric: ILD ###

"""function to calculate the intra list diversity of a given list of items (a item IDs list) using embeddings of the item-attribut-KG"""

def intra_list_diversity(input_list):
    dissimilarity = 0
    n = len(input_list)
    for i in range(n):
        for j in range(i+1, n):
            dissimilarity += distance_dict[int(input_list[i])][int(input_list[j])]
    return (2 * dissimilarity) / (n * (n - 1))

### Accuracy metrics ###


def precision(user, rec_list):
    n = len(rec_list)
    hits = 0
    relevant_items = [item for item in user_relevant_items_in_testset[user]]

    for item, predicted_rating in rec_list:
        if item in relevant_items:
            hits += 1
    return hits / n


def recall(user, rec_list):
    relevant_items = [item for item in user_relevant_items_in_testset[user]]
    n = len(relevant_items)
    hits = 0
    for item, predicted_rating in rec_list:
        if item in relevant_items:
            hits += 1
    return hits / n


def f1_mesure(p, r):
    if not (p + r) == 0:
        return 2 * p * r / (p + r)
    else:
        return 0


def average_precision(user, rec_list):
    relevant_items = [item for item in user_relevant_items_in_testset[user]]
    m = 0
    avg_p = 0
    for item, predicted_rating in rec_list:
        if item in relevant_items:
            m += 1
            item_index = rec_list.index((item, predicted_rating)) + 1 # get the index of the relevant item in rec_list
            top_index = rec_list[:item_index] # get the sub-list of rec_list up to index item_index
            p_at_index = precision(user, top_index)
            avg_p += p_at_index
    if m == 0:
        return 0
    else:
        return avg_p / m

### Objective functions ####


def f_obj(persDiv, ild_profil, candidate_item, list_rec, alpha, candidats_dissim_dict, sum_dissim, sum_accuracy):
    """objective function with two criterions, parameter persDiv determines which objective function to use"""
    div = 0
    rel = candidate_item[1]
    id_candidate_item = candidate_item[0]
    list_rec_copy = list_rec.copy()
    n = len(list_rec_copy)

    if n == 0:
        quality = (1 - alpha) * rel + alpha * div
    else:
        last_added_item = list_rec_copy[-1][0]
        div = distance_dict[last_added_item][id_candidate_item]
        candidats_dissim_dict[id_candidate_item] += div
        ild_list_rec = 2 * (candidats_dissim_dict[id_candidate_item] + sum_dissim) / (n * (n + 1))
        if persDiv == True:
            quality = (1 - alpha) * (rel + sum_accuracy) / (n + 1) - alpha * np.sqrt(square_error(ild_profil, ild_list_rec))
        else:
            quality = (1 - alpha) * (rel + sum_accuracy) / (n + 1) + alpha * ild_list_rec
    return quality


"""given a list of recommended items, compute its score of the objective function classic"""
def compute_f_obj(persDiv, ild_profil, list_rec, alpha, ild):
    accuracy = np.asarray([predicted_rating for (iid, predicted_rating) in list_rec]).sum() / len(list_rec)
    if persDiv == True:
        score = (1 - alpha) * accuracy - alpha * np.sqrt(square_error(ild_profil, ild))
    else:
        score = (1 - alpha) * accuracy + alpha * ild
    return score


"""given a user, return l(u) with each item associted with its normalised score"""
def list_u_with_score(u):
    l_u = list()
    max_score = -1 * user_candidate_items_dict[u][0][1]
    min_score = -1 * user_candidate_items_dict[u][-1][1]
    for iid, distance in user_candidate_items_dict[u]:
        s = np.divide(-1 * distance - min_score, max_score - min_score)
        l_u.append([iid, s])
    return l_u


def square_error(before, after):
    return np.power((after - before), 2)

### greedy optimization ###
"""parameters: M - number of items from which construct the list of n items, persDiv is for use which fobj"""
def bg_optimise(M, n, alpha, persDiv=False):
    diversity = 0
    precs_at_N = 0
    recalls_at_N = 0
    average_precisons = 0
    exec_time = 0
    f_obj_scores = 0
    diff_ilds = 0

    with open('results/SVD/bg_alpha'+str(alpha)+'_persDiv_'+str(persDiv)+'.csv', mode='w') as file:
        file.write('ild_profil,ild_rec\n')
        for user in users_for_test:
            ild_profil = user_profil_ild_dict[user]
            user_top_M = user_candidate_items_dict[user][:M]
            list_rec = list()
            candidats_dissim_dict = dict()
            for candidat, score in user_top_M:
                candidats_dissim_dict[candidat] = 0
            while len(list_rec) < n:
                if len(list_rec) == 0 or len(list_rec) == 1:
                    sum_dissim = 0
                else:
                    ild_rec_list = intra_list_diversity([str(item_id) for (item_id, pred_r) in list_rec])
                    sum_dissim = ild_rec_list * len(list_rec) * (len(list_rec) - 1) / 2
                sum_accuracy = np.asarray([accu for rec_i, accu in list_rec]).sum()
                best_i, best_score = get_best_item(persDiv, ild_profil, user_top_M, list_rec, alpha, candidats_dissim_dict, sum_dissim, sum_accuracy)
                list_rec.append(best_i)
                user_top_M.remove(best_i)

            for index in range(len(list_rec)):
                list_rec[index] = tuple(list_rec[index])

            ild = intra_list_diversity([str(iid) for (iid, predicted_rating) in list_rec])
            file.write(str(ild_profil)+','+str(ild)+'\n')
            diversity += ild
            diff_ilds += square_error(ild_profil, ild)

            preci = precision(user, list_rec)
            precs_at_N += preci

            reca = recall(user, list_rec)
            recalls_at_N += reca

            average_prec = average_precision(user, list_rec)
            average_precisons += average_prec

            f_obj_score = compute_f_obj(persDiv, ild_profil, list_rec, alpha, ild)
            f_obj_scores += f_obj_score

    average_diversity = round(diversity / len(users_for_test), 4)
    average_prec_at_N = round(precs_at_N / len(users_for_test), 4)
    average_recall_at_N = round(recalls_at_N / len(users_for_test), 4)
    f_mesure = round(f1_mesure(average_prec_at_N, average_recall_at_N), 4)
    mean_average_precison = round(average_precisons / len(users_for_test), 4)
    average_obj_score = round(f_obj_scores / len(users_for_test), 4)
    rmse = np.sqrt(diff_ilds / len(users_for_test))

    print("The average diversity of the BG approach is: " + str(average_diversity))
    print("The precison@N of BG approach is: " + str(average_prec_at_N))
    print("The recall@N of BG approach is: " + str(average_recall_at_N))
    print("The F-mesure@N of BG approach is: " + str(f_mesure))
    print("The MAP@N of BG approach is: " + str(mean_average_precison))
    print("The total execution time is: " + str(round(exec_time, 4)))
    print("The average score of objective function: " + str(average_obj_score))
    print("The rmse is: " + str(round(rmse, 4)))


def get_best_item(persDiv, ild_profil, list_cand, list_rec, alpha, candidats_dissim_dict, sum_dissim, sum_accuracy):
    best_score = 0
    first = True
    for candidate_item in list_cand:
        quality = f_obj(persDiv, ild_profil, candidate_item, list_rec, alpha, candidats_dissim_dict, sum_dissim, sum_accuracy)
        if first == True or quality > best_score:
            best_score = quality
            best_i = candidate_item
            first = False
    return best_i, best_score

#### Evaluation Section #####
alpha_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

print('-----------PersDiv-----------\n')
for alpha in alpha_list:
    print('Alpha is: ', alpha)
    bg_optimise(M=500, n=10, alpha=alpha, persDiv=True)
print('=============================\n')
print('=============================\n')
print('-----------Classic Div-----------\n')
for alpha in alpha_list:
    print('Alpha is: ', alpha)
    bg_optimise(M=500, n=10, alpha=alpha, persDiv=False)
