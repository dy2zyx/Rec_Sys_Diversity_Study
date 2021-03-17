import pickle
import json
import numpy as np
import pandas as pd
import collections
import math

from collections import defaultdict
from six import iteritems
from scipy.spatial.distance import cosine
from surprise import Reader, Dataset, KNNWithZScore, SVD, NMF
from matplotlib import pyplot as plt

N_OF_REC = 10
alpha_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
######################################################################
##########                                                  ##########
##########                  Load Files                      ##########
##########                                                  ##########
######################################################################
path_trainset = '../../Datasets/Book/trainset.pickle'
path_testset = '../../Datasets/Book/testset.pickle'

BASE_URI = "http://example.org/rating_ontology"
USER = BASE_URI + "/User/User_"
ITEM = BASE_URI + "/Item/Item_"
TASTE = BASE_URI + "/Taste#"

user_profil_ild_file = 'user_profil_ild_dict.pickle'
with open(user_profil_ild_file, 'rb') as user_ilds:
    user_profil_ild_dict = pickle.load(user_ilds)

sim_dict_file = 'sim_matrix_ebd.pickle'
with open(sim_dict_file, 'rb') as sim_matrix:
    distance_dict = pickle.load(sim_matrix)

print("Loading datasets...")
with open(path_trainset, 'rb') as trainset, open(path_testset, 'rb') as testset:
    trainset = pickle.load(trainset)
    testset = pickle.load(testset)

users_have_liked_items = set([user for (user, item, rating) in testset if rating > 7])
test_set = [(user, item) for (user, item, rating) in testset if rating > 7]

users_for_test = list(users_have_liked_items) # the whole set of users in testset

user_relevant_items_in_testset = defaultdict(list)

for user, item in test_set:
    user_relevant_items_in_testset[user].append(item)

print("Datasets loaded.")

######################################################################
##########                                                  ##########
##########                   Metrics                        ##########
##########                                                  ##########
######################################################################


def intra_list_diversity(input_list):
    """function to calculate the intra list diversity of a given list of items (a item IDs list)
    using embeddings of the item-attribut-KG"""
    dissimilarity = 0
    n = len(input_list)
    for i in range(n):
        for j in range(i+1, n):
            dissimilarity += distance_dict[int(input_list[i])][int(input_list[j])]
    return (2 * dissimilarity) / (n * (n - 1))


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


######################################################################
##########                                                  ##########
##########                  Objective Functions             ##########
##########                                                  ##########
######################################################################
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


def square_error(before, after):
    return np.power((after - before), 2)

######################################################################
##########                                                  ##########
##########                        CBF                       ##########
##########                                                  ##########
######################################################################
print("Computation starts...")
print("CBF...")


path_user_candidate_items = 'candidates_dicts/user_candidate_items_dict_cbf.pickle'
"""user_candidate_items_dict: uid (int) -> list of candidates items ordered by accuracy"""
with open(path_user_candidate_items, 'rb') as user_candidate_items:
    user_candidate_items_dict_cbf = pickle.load(user_candidate_items)


def bg_optimise_cbf(M, n, alpha, persDiv, candidate_items_dict):
    algo = "CBF"
    objective_func = "personalized" if persDiv else "classic"
    diversity = 0
    precs_at_N = 0
    recalls_at_N = 0
    average_precisons = 0
    diff_ilds = 0

    with open('results/CBF/bg_alpha'+str(alpha)+'_persDiv_'+str(persDiv)+'.csv', mode='w') as file:
        file.write('ild_profil,ild_rec\n')

        for user in users_for_test:
            ild_profil = user_profil_ild_dict[user]
            user_top_M = candidate_items_dict[user][:M]
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

    average_diversity = round(diversity / len(users_for_test), 4)
    average_prec_at_N = round(precs_at_N / len(users_for_test), 4)
    average_recall_at_N = round(recalls_at_N / len(users_for_test), 4)
    f_mesure = round(f1_mesure(average_prec_at_N, average_recall_at_N), 4)
    mean_average_precison = round(average_precisons / len(users_for_test), 4)
    rmse = round(np.sqrt(diff_ilds / len(users_for_test)), 4)

    accu_cbf.write(str(alpha) + " " +
                objective_func + " " +
                str(average_diversity) + " " +
                str(average_prec_at_N) + " " +
                str(average_recall_at_N) + " " +
                str(f_mesure) + " " +
                str(mean_average_precison) + " " +
                str(rmse) + " " +
                algo + "\n"
                )


with open('results/CBF/accuracy_cbf.csv', 'w') as accu_cbf:
    accu_cbf.write('alpha f_obj ILD Precision Recall F1_mesure MAP rmsde algo\n')
    print('-----------PersDiv-----------\n')
    for alpha in alpha_list:
        print('Alpha is: ', alpha)
        bg_optimise_cbf(M=200, n=N_OF_REC, alpha=alpha, persDiv=True, candidate_items_dict=user_candidate_items_dict_cbf)
    print('=============================\n')
    print('-----------Classic Div-----------\n')
    for alpha in alpha_list:
        print('Alpha is: ', alpha)
        bg_optimise_cbf(M=200, n=N_OF_REC, alpha=alpha, persDiv=False, candidate_items_dict=user_candidate_items_dict_cbf)


######################################################################
##########                                                  ##########
##########                    TopPopular                    ##########
##########                                                  ##########
######################################################################
print("TopPopular...")

path_user_candidate_items = 'candidates_dicts/user_candidate_items_dict_top_pop.pickle'
"""user_candidate_items_dict: uid (int) -> list of candidates items ordered by accuracy"""
with open(path_user_candidate_items, 'rb') as user_candidate_items:
    user_candidate_items_dict_top_pop = pickle.load(user_candidate_items)


def bg_optimise_top_pop(M, n, alpha, persDiv, candidate_items_dict):
    algo = "TopPopular"
    objective_func = "personalized" if persDiv else "classic"
    diversity = 0
    precs_at_N = 0
    recalls_at_N = 0
    average_precisons = 0
    diff_ilds = 0

    with open('results/TopPopular/bg_alpha'+str(alpha)+'_persDiv_'+str(persDiv)+'.csv', mode='w') as file:
        file.write('ild_profil,ild_rec\n')

        for user in users_for_test:
            ild_profil = user_profil_ild_dict[user]
            user_top_M = candidate_items_dict[user][:M]
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

    average_diversity = round(diversity / len(users_for_test), 4)
    average_prec_at_N = round(precs_at_N / len(users_for_test), 4)
    average_recall_at_N = round(recalls_at_N / len(users_for_test), 4)
    f_mesure = round(f1_mesure(average_prec_at_N, average_recall_at_N), 4)
    mean_average_precison = round(average_precisons / len(users_for_test), 4)
    rmse = round(np.sqrt(diff_ilds / len(users_for_test)), 4)

    accu_top_pop.write(str(alpha) + " " +
                objective_func + " " +
                str(average_diversity) + " " +
                str(average_prec_at_N) + " " +
                str(average_recall_at_N) + " " +
                str(f_mesure) + " " +
                str(mean_average_precison) + " " +
                str(rmse) + " " +
                algo + "\n"
                )


with open('results/TopPopular/accuracy_top_pop.csv', 'w') as accu_top_pop:
    accu_top_pop.write('alpha f_obj ILD Precision Recall F1_mesure MAP rmsde algo\n')
    print('-----------PersDiv-----------\n')
    for alpha in alpha_list:
        print('Alpha is: ', alpha)
        bg_optimise_top_pop(M=200, n=N_OF_REC, alpha=alpha, persDiv=True, candidate_items_dict=user_candidate_items_dict_top_pop)
    print('=============================\n')
    print('-----------Classic Div-----------\n')
    for alpha in alpha_list:
        print('Alpha is: ', alpha)
        bg_optimise_top_pop(M=200, n=N_OF_REC, alpha=alpha, persDiv=False, candidate_items_dict=user_candidate_items_dict_top_pop)


######################################################################
##########                                                  ##########
##########                    CBF-TopPopular                ##########
##########                                                  ##########
######################################################################
print("CBF-TopPopular...")

path_user_candidate_items = 'candidates_dicts/user_candidate_items_dict_cbf_pop.pickle'
"""user_candidate_items_dict: uid (int) -> list of candidates items ordered by accuracy"""
with open(path_user_candidate_items, 'rb') as user_candidate_items:
    user_candidate_items_dict_cbf_pop = pickle.load(user_candidate_items)


def bg_optimise_cbf_pop(M, n, alpha, persDiv, candidate_items_dict):
    algo = "CBF-TopPopular"
    objective_func = "personalized" if persDiv else "classic"
    diversity = 0
    precs_at_N = 0
    recalls_at_N = 0
    average_precisons = 0
    diff_ilds = 0

    with open('results/CBF_TopPopular/bg_alpha'+str(alpha)+'_persDiv_'+str(persDiv)+'.csv', mode='w') as file:
        file.write('ild_profil,ild_rec\n')

        for user in users_for_test:
            ild_profil = user_profil_ild_dict[user]
            user_top_M = candidate_items_dict[user][:M]
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

    average_diversity = round(diversity / len(users_for_test), 4)
    average_prec_at_N = round(precs_at_N / len(users_for_test), 4)
    average_recall_at_N = round(recalls_at_N / len(users_for_test), 4)
    f_mesure = round(f1_mesure(average_prec_at_N, average_recall_at_N), 4)
    mean_average_precison = round(average_precisons / len(users_for_test), 4)
    rmse = round(np.sqrt(diff_ilds / len(users_for_test)), 4)

    accu_cbf_pop.write(str(alpha) + " " +
                objective_func + " " +
                str(average_diversity) + " " +
                str(average_prec_at_N) + " " +
                str(average_recall_at_N) + " " +
                str(f_mesure) + " " +
                str(mean_average_precison) + " " +
                str(rmse) + " " +
                algo + "\n"
                )


with open('results/CBF_TopPopular/accuracy_cbf_pop.csv', 'w') as accu_cbf_pop:
    accu_cbf_pop.write('alpha f_obj ILD Precision Recall F1_mesure MAP rmsde algo\n')
    print('-----------PersDiv-----------\n')
    for alpha in alpha_list:
        print('Alpha is: ', alpha)
        bg_optimise_cbf_pop(M=200, n=N_OF_REC, alpha=alpha, persDiv=True, candidate_items_dict=user_candidate_items_dict_cbf_pop)
    print('=============================\n')
    print('-----------Classic Div-----------\n')
    for alpha in alpha_list:
        print('Alpha is: ', alpha)
        bg_optimise_cbf_pop(M=200, n=N_OF_REC, alpha=alpha, persDiv=False, candidate_items_dict=user_candidate_items_dict_cbf_pop)

######################################################################
##########                                                  ##########
##########                    IBCF                          ##########
##########                                                  ##########
######################################################################
print("IBCF...")

path_user_candidate_items = 'candidates_dicts/user_candidate_items_dict_ibcf.pickle'
"""user_candidate_items_dict: uid (int) -> list of candidates items ordered by accuracy"""
with open(path_user_candidate_items, 'rb') as user_candidate_items:
    user_candidate_items_dict_ibcf = pickle.load(user_candidate_items)


def bg_optimise_ibcf(M, n, alpha, persDiv, candidate_items_dict):
    algo = "IBCF"
    objective_func = "personalized" if persDiv else "classic"
    diversity = 0
    precs_at_N = 0
    recalls_at_N = 0
    average_precisons = 0
    diff_ilds = 0

    with open('results/IBCF/bg_alpha'+str(alpha)+'_persDiv_'+str(persDiv)+'.csv', mode='w') as file:
        file.write('ild_profil,ild_rec\n')

        for user in users_for_test:
            ild_profil = user_profil_ild_dict[user]
            user_top_M = candidate_items_dict[user][:M]
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

    average_diversity = round(diversity / len(users_for_test), 4)
    average_prec_at_N = round(precs_at_N / len(users_for_test), 4)
    average_recall_at_N = round(recalls_at_N / len(users_for_test), 4)
    f_mesure = round(f1_mesure(average_prec_at_N, average_recall_at_N), 4)
    mean_average_precison = round(average_precisons / len(users_for_test), 4)
    rmse = round(np.sqrt(diff_ilds / len(users_for_test)), 4)

    accu_ibcf.write(str(alpha) + " " +
                objective_func + " " +
                str(average_diversity) + " " +
                str(average_prec_at_N) + " " +
                str(average_recall_at_N) + " " +
                str(f_mesure) + " " +
                str(mean_average_precison) + " " +
                str(rmse) + " " +
                algo + "\n"
                )


with open('results/IBCF/accuracy_ibcf.csv', 'w') as accu_ibcf:
    accu_ibcf.write('alpha f_obj ILD Precision Recall F1_mesure MAP rmsde algo\n')
    print('-----------PersDiv-----------\n')
    for alpha in alpha_list:
        print('Alpha is: ', alpha)
        bg_optimise_ibcf(M=200, n=N_OF_REC, alpha=alpha, persDiv=True, candidate_items_dict=user_candidate_items_dict_ibcf)
    print('=============================\n')
    print('-----------Classic Div-----------\n')
    for alpha in alpha_list:
        print('Alpha is: ', alpha)
        bg_optimise_ibcf(M=200, n=N_OF_REC, alpha=alpha, persDiv=False, candidate_items_dict=user_candidate_items_dict_ibcf)


######################################################################
##########                                                  ##########
##########                    SVD                          ##########
##########                                                  ##########
######################################################################
print("SVD...")

path_user_candidate_items = 'candidates_dicts/user_candidate_items_dict_svd.pickle'
"""user_candidate_items_dict: uid (int) -> list of candidates items ordered by accuracy"""
with open(path_user_candidate_items, 'rb') as user_candidate_items:
    user_candidate_items_dict_svd = pickle.load(user_candidate_items)


def bg_optimise_svd(M, n, alpha, persDiv, candidate_items_dict):
    algo = "SVD"
    objective_func = "personalized" if persDiv else "classic"
    diversity = 0
    precs_at_N = 0
    recalls_at_N = 0
    average_precisons = 0
    diff_ilds = 0

    with open('results/SVD/bg_alpha'+str(alpha)+'_persDiv_'+str(persDiv)+'.csv', mode='w') as file:
        file.write('ild_profil,ild_rec\n')

        for user in users_for_test:
            ild_profil = user_profil_ild_dict[user]
            user_top_M = candidate_items_dict[user][:M]
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

    average_diversity = round(diversity / len(users_for_test), 4)
    average_prec_at_N = round(precs_at_N / len(users_for_test), 4)
    average_recall_at_N = round(recalls_at_N / len(users_for_test), 4)
    f_mesure = round(f1_mesure(average_prec_at_N, average_recall_at_N), 4)
    mean_average_precison = round(average_precisons / len(users_for_test), 4)
    rmse = round(np.sqrt(diff_ilds / len(users_for_test)), 4)

    accu_svd.write(str(alpha) + " " +
                objective_func + " " +
                str(average_diversity) + " " +
                str(average_prec_at_N) + " " +
                str(average_recall_at_N) + " " +
                str(f_mesure) + " " +
                str(mean_average_precison) + " " +
                str(rmse) + " " +
                algo + "\n"
                )


with open('results/SVD/accuracy_svd.csv', 'w') as accu_svd:
    accu_svd.write('alpha f_obj ILD Precision Recall F1_mesure MAP rmsde algo\n')
    print('-----------PersDiv-----------\n')
    for alpha in alpha_list:
        print('Alpha is: ', alpha)
        bg_optimise_svd(M=200, n=N_OF_REC, alpha=alpha, persDiv=True, candidate_items_dict=user_candidate_items_dict_svd)
    print('=============================\n')
    print('-----------Classic Div-----------\n')
    for alpha in alpha_list:
        print('Alpha is: ', alpha)
        bg_optimise_svd(M=200, n=N_OF_REC, alpha=alpha, persDiv=False, candidate_items_dict=user_candidate_items_dict_svd)


######################################################################
##########                                                  ##########
##########                    DNN                           ##########
##########                                                  ##########
######################################################################
print("DNN...")

path_user_candidate_items = 'candidates_dicts/user_candidate_items_dict_dnn.pickle'
"""user_candidate_items_dict: uid (int) -> list of candidates items ordered by accuracy"""
with open(path_user_candidate_items, 'rb') as user_candidate_items:
    user_candidate_items_dict_dnn = pickle.load(user_candidate_items)


def bg_optimise_dnn(M, n, alpha, persDiv, candidate_items_dict):
    algo = "DNN"
    objective_func = "personalized" if persDiv else "classic"
    diversity = 0
    precs_at_N = 0
    recalls_at_N = 0
    average_precisons = 0
    diff_ilds = 0

    with open('results/DNN/bg_alpha'+str(alpha)+'_persDiv_'+str(persDiv)+'.csv', mode='w') as file:
        file.write('ild_profil,ild_rec\n')

        for user in users_for_test:
            ild_profil = user_profil_ild_dict[user]
            user_top_M = candidate_items_dict[user][:M]
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

    average_diversity = round(diversity / len(users_for_test), 4)
    average_prec_at_N = round(precs_at_N / len(users_for_test), 4)
    average_recall_at_N = round(recalls_at_N / len(users_for_test), 4)
    f_mesure = round(f1_mesure(average_prec_at_N, average_recall_at_N), 4)
    mean_average_precison = round(average_precisons / len(users_for_test), 4)
    rmse = round(np.sqrt(diff_ilds / len(users_for_test)), 4)

    accu_dnn.write(str(alpha) + " " +
                objective_func + " " +
                str(average_diversity) + " " +
                str(average_prec_at_N) + " " +
                str(average_recall_at_N) + " " +
                str(f_mesure) + " " +
                str(mean_average_precison) + " " +
                str(rmse) + " " +
                algo + "\n"
                )


with open('results/DNN/accuracy_dnn.csv', 'w') as accu_dnn:
    accu_dnn.write('alpha f_obj ILD Precision Recall F1_mesure MAP rmsde algo\n')
    print('-----------PersDiv-----------\n')
    for alpha in alpha_list:
        print('Alpha is: ', alpha)
        bg_optimise_dnn(M=200, n=N_OF_REC, alpha=alpha, persDiv=True, candidate_items_dict=user_candidate_items_dict_dnn)
    print('=============================\n')
    print('-----------Classic Div-----------\n')
    for alpha in alpha_list:
        print('Alpha is: ', alpha)
        bg_optimise_dnn(M=200, n=N_OF_REC, alpha=alpha, persDiv=False, candidate_items_dict=user_candidate_items_dict_dnn)


######################################################################
##########                                                  ##########
##########                    KGE                           ##########
##########                                                  ##########
######################################################################
print("KGE...")

path_user_candidate_items = 'candidates_dicts/user_candidate_items_dict_kge.pickle'
"""user_candidate_items_dict: uid (int) -> list of candidates items ordered by accuracy"""
with open(path_user_candidate_items, 'rb') as user_candidate_items:
    user_candidate_items_dict_kge = pickle.load(user_candidate_items)


def bg_optimise_kge(M, n, alpha, persDiv, candidate_items_dict):
    algo = "KGE"
    objective_func = "personalized" if persDiv else "classic"
    diversity = 0
    precs_at_N = 0
    recalls_at_N = 0
    average_precisons = 0
    diff_ilds = 0

    with open('results/KGE/bg_alpha'+str(alpha)+'_persDiv_'+str(persDiv)+'.csv', mode='w') as file:
        file.write('ild_profil,ild_rec\n')

        for user in users_for_test:
            ild_profil = user_profil_ild_dict[user]
            user_top_M = candidate_items_dict[user][:M]
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

    average_diversity = round(diversity / len(users_for_test), 4)
    average_prec_at_N = round(precs_at_N / len(users_for_test), 4)
    average_recall_at_N = round(recalls_at_N / len(users_for_test), 4)
    f_mesure = round(f1_mesure(average_prec_at_N, average_recall_at_N), 4)
    mean_average_precison = round(average_precisons / len(users_for_test), 4)
    rmse = round(np.sqrt(diff_ilds / len(users_for_test)), 4)

    accu_kge.write(str(alpha) + " " +
                objective_func + " " +
                str(average_diversity) + " " +
                str(average_prec_at_N) + " " +
                str(average_recall_at_N) + " " +
                str(f_mesure) + " " +
                str(mean_average_precison) + " " +
                str(rmse) + " " +
                algo + "\n"
                )


with open('results/KGE/accuracy_kge.csv', 'w') as accu_kge:
    accu_kge.write('alpha f_obj ILD Precision Recall F1_mesure MAP rmsde algo\n')
    print('-----------PersDiv-----------\n')
    for alpha in alpha_list:
        print('Alpha is: ', alpha)
        bg_optimise_kge(M=200, n=N_OF_REC, alpha=alpha, persDiv=True, candidate_items_dict=user_candidate_items_dict_kge)
    print('=============================\n')
    print('-----------Classic Div-----------\n')
    for alpha in alpha_list:
        print('Alpha is: ', alpha)
        bg_optimise_kge(M=200, n=N_OF_REC, alpha=alpha, persDiv=False, candidate_items_dict=user_candidate_items_dict_kge)