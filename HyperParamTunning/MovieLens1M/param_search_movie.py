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
import pykeen
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

from pykeen.kge_models import TransE
from collections import defaultdict
from six import iteritems
from scipy.spatial.distance import cosine
from surprise import Reader, Dataset, SVD, KNNWithZScore
from bayes_opt import BayesianOptimization

######################################################################
##########                                                  ##########
##########                  Load Datasets                   ##########
##########                                                  ##########
######################################################################

work_dir = ''
N_ITERATION_BO = 20

with open(work_dir + 'trainset.pickle', 'rb') as f1, open(work_dir + 'validation_set.pickle', 'rb') as f2:
    trainset = pickle.load(f1)
    validation_set = pickle.load(f2)

users_have_liked_items = set([user for (user, item, rating) in validation_set if rating > 3])

validation_set = [(user, item) for (user, item, rating) in validation_set if rating > 3]
users_for_validate = list(users_have_liked_items)

print(len(users_for_validate))

user_relevant_items_in_validation = defaultdict(list)

for user, item in validation_set:
    user_relevant_items_in_validation[user].append(item)

######
user_items_in_validation_set = defaultdict(list)
items_in_validation_set = set()
items_in_trainset = set()
user_rated_items = defaultdict(list)
user_unrated_items = defaultdict(list)

for user, item in validation_set:
    user_items_in_validation_set[user].append(item)
    items_in_validation_set.add(item)

for (user, item, rating) in trainset:
    user_rated_items[user].append(item)
    items_in_trainset.add(item)

all_possible_items = set(list(items_in_validation_set) + list(items_in_trainset))

for user in user_items_in_validation_set.keys():
    un_rated_items = [item for item in all_possible_items if item not in user_rated_items[user]]
    user_unrated_items[user] = un_rated_items


######

######################################################################
##########                                                  ##########
##########                  Define Optim Functions          ##########
##########                                                  ##########
######################################################################


def precision(user, rec_list):
    n = len(rec_list)
    hits = 0
    relevant_items = [item for item in user_relevant_items_in_validation[user]]

    for item, predicted_rating in rec_list:
        if item in relevant_items:
            hits += 1
    return hits / n


def average_precision(user, rec_list):
    relevant_items = [item for item in user_relevant_items_in_validation[user]]
    m = 0
    avg_p = 0
    for item, predicted_rating in rec_list:
        if item in relevant_items:
            m += 1
            item_index = rec_list.index((item, predicted_rating)) + 1
            top_index = rec_list[:item_index]
            p_at_index = precision(user, top_index)
            avg_p += p_at_index
    if m == 0:
        return 0
    else:
        return avg_p / m


######################################################################
##########                                                  ##########
##########                  Tune CBF-MostPop                ##########
##########                                                  ##########
######################################################################
# print("Tune CBF-MostPop")
# # popularity part
# item_popularity_dict = defaultdict(list)
# for u, i, r in trainset:
#     item_popularity_dict[i].append(u)

# popu_max = max([len(item_popularity_dict[i]) for i in item_popularity_dict.keys()])
# popu_min = min([len(item_popularity_dict[i]) for i in item_popularity_dict.keys()])

# item_popularity_dict_2 = dict()
# for item in item_popularity_dict.keys():
#     item_popu = (len(item_popularity_dict[item]) - popu_min) / (popu_max - popu_min)
#     item_popularity_dict_2[item] = item_popu

# # cbf part
# sim_dict_file = '/Users/yudu/Documents/phd/dataset/Movie/diversity_purpose/KG_rec/ExpansionApproach/sim_matrix_ebd.pickle'
# with open(sim_dict_file, 'rb') as sim_matrix:
#     distance_dict = pickle.load(sim_matrix)

# user_positive_profils = defaultdict(list)

# for user, item, rating in trainset:
#     if rating > 3:
#         user_positive_profils[user].append(item)


# def predict_rating(user, item):
#     length_profil = len(user_positive_profils[user])
#     if length_profil == 0:
#         return 0
#     else:
#         rating = 0
#         for i in user_positive_profils[user]:
#             sim_i_item = 1 - distance_dict[i][item]
#             rating += sim_i_item
#         return rating / length_profil


# def recommend_topN_Mostpop_CBF(user, weight):
#     top_n = list()
#     for un_rated_item in user_unrated_items[user]:
#         item_popularity = item_popularity_dict_2[un_rated_item] if un_rated_item in item_popularity_dict_2.keys() else 0
#         cbf_score = predict_rating(user, un_rated_item)
#         score = weight * item_popularity + (1 - weight) * cbf_score
#         top_n.append((un_rated_item, score))
#     top_n.sort(key=lambda x: x[1], reverse=True)
#     return top_n[:10]


# tuning_params_cbf_pop = {
#     'weight': (0, 1)
# }


# def optim_func_cbf_pop(weight):
#     MAP = 0
#     nb_users = len(users_for_validate)
#     for user in users_for_validate:
#         top_n_list = recommend_topN_Mostpop_CBF(user, weight)
#         ap_user = average_precision(user, top_n_list)
#         MAP += ap_user
#     MAP = MAP / nb_users
#     return MAP


# optimizer_cbf_pop = BayesianOptimization(
#     f=optim_func_cbf_pop,
#     pbounds=tuning_params_cbf_pop,
#     verbose=0,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
#     random_state=1,
# )

# logger = JSONLogger(path="CBF-TopPopular/logs_cbf_pop_PO.json")
# optimizer_cbf_pop.subscribe(Events.OPTIMIZATION_STEP, logger)

# optimizer_cbf_pop.maximize(
#     init_points=0,
#     n_iter=N_ITERATION_BO,
# )

######################################################################
##########                                                  ##########
##########                  Tune UBCF                       ##########
##########                                                  ##########
######################################################################
print("Tune UBCF")
data = pd.DataFrame([(u, i, r) for u, i, r in trainset], columns=['userID', 'itemID', 'rating'])
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(data, reader)

train_set = dataset.build_full_trainset()


def recommend_topN_cf(user, model):
    top_n = list()
    for un_rated_item in user_unrated_items[user]:
        predicted_rating = model.predict(user, un_rated_item)[3]
        top_n.append((un_rated_item, predicted_rating))
    top_n.sort(key=lambda x: x[1], reverse=True)
    return top_n[:10]


tuning_params_cf = {
    "k": (1, 200),
}


def optim_func_cf(k):
    k = int(k)
    cf_model = KNNWithZScore(k=k, sim_options={'name': 'cosine', 'user_based': False}, verbose=False)
    cf_model.fit(train_set)
    MAP = 0
    nb_users = len(users_for_validate)
    for user in users_for_validate:
        top_n_list = recommend_topN_cf(user, cf_model)
        ap_user = average_precision(user, top_n_list)
        MAP += ap_user
    MAP = MAP / nb_users
    return MAP


optimizer_cf = BayesianOptimization(
    f=optim_func_cf,
    pbounds=tuning_params_cf,
    verbose=0,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

logger = JSONLogger(path="IBCF/logs_ibcf_PO3.json")
optimizer_cf.subscribe(Events.OPTIMIZATION_STEP, logger)

optimizer_cf.maximize(
    init_points=1,
    n_iter=N_ITERATION_BO,
)


######################################################################
##########                                                  ##########
##########                  Tune SVD                        ##########
##########                                                  ##########
######################################################################
print("Tune SVD")
def recommend_topN_svd(user, model):
    top_n = list()
    for un_rated_item in user_unrated_items[user]:
        predicted_rating = model.predict(user, un_rated_item)[3]
        top_n.append((un_rated_item, predicted_rating))
    top_n.sort(key=lambda x: x[1], reverse=True)
    return top_n[:10]


tuning_params_svd = {
    "n_factors": (50, 200),
    "n_epochs": (1, 100),
    "lr_all": (0, 0.1),
    "reg_all": (0, 0.1)
}


def optim_func_svd(n_factors, n_epochs, lr_all, reg_all):
    n_factors = int(n_factors)
    n_epochs = int(n_epochs)
    svd = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
    svd.fit(train_set)
    MAP = 0
    nb_users = len(users_for_validate)
    for user in users_for_validate:
        top_n_list = recommend_topN_svd(user, svd)
        ap_user = average_precision(user, top_n_list)
        MAP += ap_user
    MAP = MAP / nb_users
    return MAP


optimizer_svd = BayesianOptimization(
    f=optim_func_svd,
    pbounds=tuning_params_svd,
    verbose=0,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

logger = JSONLogger(path="SVD/logs_svd_PO3.json")
optimizer_svd.subscribe(Events.OPTIMIZATION_STEP, logger)

optimizer_svd.maximize(
    init_points=1,
    n_iter=N_ITERATION_BO,
)

######################################################################
##########                                                  ##########
##########                  Tune KGE                        ##########
##########                                                  ##########
######################################################################
# print("Tune KGE")
# BASE_URI = "http://example.org/rating_ontology"
# USER_ = BASE_URI + "/User/User_"
# ITEM_ = BASE_URI + "/Item/Item_"
# TASTE_ = BASE_URI + "/Taste#"
#
# """function to recommend the topN items for a user given the rating prediction model"""
#
#
# def recommend_topN_kge(user, dict_entity_embedding, dict_relation_embedding):
#     top_n = list()
#     for un_rated_item in user_unrated_items[user]:
#         user_embedding = dict_entity_embedding[USER_ + str(user)]
#         r_like_embedding = dict_relation_embedding[TASTE_ + 'like']
#         item_embedding = dict_entity_embedding[ITEM_ + str(un_rated_item)]
#         score = compute_score(user_embedding, r_like_embedding, item_embedding)
#         top_n.append((un_rated_item, score))
#     top_n.sort(key=lambda x: x[1], reverse=True)
#     return top_n[:10]
#
#
# def compute_score(h_embedding, r_embedding, t_embedding):
#     sum_vec = np.asarray(h_embedding) + np.asarray(r_embedding) - np.asarray(t_embedding)
#     distance_l1 = np.linalg.norm(sum_vec, ord=1)
#     return -distance_l1
#
#
# tuning_params_kge = {
#     "embedding_dim": (10, 200),
#     "margin_loss": (1, 10),
#     "lr": (0.001, 0.1),
#     "batch_size": (16, 256),
#     "num_epochs": (100, 1000)
# }
#
#
# def optim_func_kge(embedding_dim, margin_loss, lr, batch_size, num_epochs):
#     embedding_dim = int(embedding_dim)
#     margin_loss = int(margin_loss)
#     batch_size = int(batch_size)
#     num_epochs = int(num_epochs)
#     print(embedding_dim, margin_loss, batch_size, num_epochs)
#
#     output_directory = 'KGE/train_model/'
#     config = dict(
#         training_set_path='Files/kg_train.nt',
#         execution_mode='Training_mode',
#         kg_embedding_model_name='TransE',
#         embedding_dim=embedding_dim,
#         normalization_of_entities=2,  # corresponds to L2
#         scoring_function=1,  # corresponds to L1
#         margin_loss=margin_loss,
#         learning_rate=lr,
#         batch_size=batch_size,
#         num_epochs=num_epochs,
#         filter_negative_triples=True,
#         random_seed=0,
#         preferred_device='gpu',
#     )
#     print(config)
#     results = pykeen.run(
#         config=config,
#         output_directory=output_directory,
#     )
#
#     dict_entity_embedding = list(results.results.values())[2]
#     dict_relation_embedding = list(results.results.values())[3]
#     MAP = 0
#     nb_users = len(users_for_validate)
#     for user in users_for_validate:
#         top_n_list = recommend_topN_kge(user, dict_entity_embedding, dict_relation_embedding)
#         ap_user = average_precision(user, top_n_list)
#         MAP += ap_user
#     MAP = MAP / nb_users
#     return MAP
#
# optimizer_kge = BayesianOptimization(
#     f=optim_func_kge,
#     pbounds=tuning_params_kge,
#     verbose=0, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
#     random_state=1,
# )
#
# logger = JSONLogger(path="KGE/logs_kge_PO.json")
# optimizer_kge.subscribe(Events.OPTIMIZATION_STEP, logger)
#
# optimizer_kge.maximize(
#     init_points=1,
#     n_iter=N_ITERATION_BO
# )
#
# ######################################################################
# ##########                                                  ##########
# ##########                  Tune DNN                        ##########
# ##########                                                  ##########
# ######################################################################
# print("Tune DNN")
# train_rating = 'DNN/train.csv'
# config_model = 'NeuMF-end'
#
#
# def recommend_topN_dnn(user, model):
#     top_n = list()
#     for un_rated_item in user_unrated_items[user]:
#         predicted_rating = model(torch.tensor([[user]]).cuda(), torch.tensor([[un_rated_item]]).cuda())
#         predicted_rating = predicted_rating.item()
#
#         top_n.append((un_rated_item, predicted_rating))
#     top_n.sort(key=lambda x:x[1], reverse=True)
#     return top_n[:10]
#
#
# class NCFData(data.Dataset):
#     def __init__(self, features,
#                  num_item, train_mat=None, num_ng=0, is_training=None):
#         super(NCFData, self).__init__()
#         """ Note that the labels are only useful when training, we thus
# 			add them in the ng_sample() function.
# 		"""
#         self.features_ps = features
#         self.num_item = num_item
#         self.train_mat = train_mat
#         self.num_ng = num_ng
#         self.is_training = is_training
#         self.labels = [0 for _ in range(len(features))]
#
#     def ng_sample(self):
#         assert self.is_training, 'no need to sampling when testing'
#
#         self.features_ng = []
#         for x in self.features_ps:
#             u = x[0]
#             for t in range(self.num_ng):
#                 j = np.random.randint(self.num_item)
#                 while (u, j) in self.train_mat:
#                     j = np.random.randint(self.num_item)
#                 self.features_ng.append([u, j])
#
#         labels_ps = [1 for _ in range(len(self.features_ps))]
#         labels_ng = [0 for _ in range(len(self.features_ng))]
#
#         self.features_fill = self.features_ps + self.features_ng
#         self.labels_fill = labels_ps + labels_ng
#
#     def __len__(self):
#         return (self.num_ng + 1) * len(self.labels)
#
#     def __getitem__(self, idx):
#         features = self.features_fill if self.is_training \
#             else self.features_ps
#         labels = self.labels_fill if self.is_training \
#             else self.labels
#
#         user = features[idx][0]
#         item = features[idx][1]
#         label = labels[idx]
#         return user, item, label
#
#
# class NCF(nn.Module):
#     def __init__(self, user_num, item_num, factor_num, num_layers,
#                  dropout, model, GMF_model=None, MLP_model=None):
#         super(NCF, self).__init__()
#
#         self.dropout = dropout
#         self.model = model
#         self.GMF_model = GMF_model
#         self.MLP_model = MLP_model
#
#         self.embed_user_GMF = nn.Embedding(user_num, factor_num)
#         self.embed_item_GMF = nn.Embedding(item_num, factor_num)
#         self.embed_user_MLP = nn.Embedding(
#             user_num, factor_num * (2 ** (num_layers - 1)))
#         self.embed_item_MLP = nn.Embedding(
#             item_num, factor_num * (2 ** (num_layers - 1)))
#
#         MLP_modules = []
#         for i in range(num_layers):
#             input_size = factor_num * (2 ** (num_layers - i))
#             MLP_modules.append(nn.Dropout(p=self.dropout))
#             MLP_modules.append(nn.Linear(input_size, input_size // 2))
#             MLP_modules.append(nn.ReLU())
#         self.MLP_layers = nn.Sequential(*MLP_modules)
#
#         if self.model in ['MLP', 'GMF']:
#             predict_size = factor_num
#         else:
#             predict_size = factor_num * 2
#         self.predict_layer = nn.Linear(predict_size, 1)
#
#         self._init_weight_()
#
#     def _init_weight_(self):
#         """ We leave the weights initialization here. """
#         if not self.model == 'NeuMF-pre':
#             nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
#             nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
#             nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
#             nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
#
#             for m in self.MLP_layers:
#                 if isinstance(m, nn.Linear):
#                     nn.init.xavier_uniform_(m.weight)
#             nn.init.kaiming_uniform_(self.predict_layer.weight,
#                                      a=1, nonlinearity='sigmoid')
#
#             for m in self.modules():
#                 if isinstance(m, nn.Linear) and m.bias is not None:
#                     m.bias.data.zero_()
#         else:
#             # embedding layers
#             self.embed_user_GMF.weight.data.copy_(
#                 self.GMF_model.embed_user_GMF.weight)
#             self.embed_item_GMF.weight.data.copy_(
#                 self.GMF_model.embed_item_GMF.weight)
#             self.embed_user_MLP.weight.data.copy_(
#                 self.MLP_model.embed_user_MLP.weight)
#             self.embed_item_MLP.weight.data.copy_(
#                 self.MLP_model.embed_item_MLP.weight)
#
#             # mlp layers
#             for (m1, m2) in zip(
#                     self.MLP_layers, self.MLP_model.MLP_layers):
#                 if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
#                     m1.weight.data.copy_(m2.weight)
#                     m1.bias.data.copy_(m2.bias)
#
#             # predict layers
#             predict_weight = torch.cat([
#                 self.GMF_model.predict_layer.weight,
#                 self.MLP_model.predict_layer.weight], dim=1)
#             precit_bias = self.GMF_model.predict_layer.bias + \
#                           self.MLP_model.predict_layer.bias
#
#             self.predict_layer.weight.data.copy_(0.5 * predict_weight)
#             self.predict_layer.bias.data.copy_(0.5 * precit_bias)
#
#     def forward(self, user, item):
#         if not self.model == 'MLP':
#             embed_user_GMF = self.embed_user_GMF(user)
#             embed_item_GMF = self.embed_item_GMF(item)
#             output_GMF = embed_user_GMF * embed_item_GMF
#         if not self.model == 'GMF':
#             embed_user_MLP = self.embed_user_MLP(user)
#             embed_item_MLP = self.embed_item_MLP(item)
#             interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
#             output_MLP = self.MLP_layers(interaction)
#
#         if self.model == 'GMF':
#             concat = output_GMF
#         elif self.model == 'MLP':
#             concat = output_MLP
#         else:
#             concat = torch.cat((output_GMF, output_MLP), -1)
#
#         prediction = self.predict_layer(concat)
#         return prediction.view(-1)
#
#
# def load_all(test_num=100):
#     """ We load all the three file here to save time in each epoch. """
#     train_data = pd.read_csv(
#         train_rating,
#         sep='\t', header=None, names=['user', 'item'],
#         usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
#
#     user_num = train_data['user'].max() + 1
#     item_num = train_data['item'].max() + 1
#
#     train_data = train_data.values.tolist()
#
#     # load ratings as a dok matrix
#     train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
#     for x in train_data:
#         train_mat[x[0], x[1]] = 1.0
#
#     test_data = []
#     # with open(test_negative, 'r') as fd:
#     # 	line = fd.readline()
#     # 	while line != None and line != '':
#     # 		arr = line.split('\t')
#     # 		u = eval(arr[0])[0]
#     # 		test_data.append([u, eval(arr[0])[1]])
#     # 		for i in arr[1:]:
#     # 			test_data.append([u, int(i)])
#     # 		line = fd.readline()
#     return train_data, test_data, user_num, item_num, train_mat
#
#
# def optim_func_dnn(batch_size, factor_num, num_layers, dropout, lr, epochs, num_ng):
#     batch_size = int(batch_size)
#     factor_num = int(factor_num)
#     num_layers = int(num_layers)
#     epochs = int(epochs)
#     num_ng = int(num_ng)
#     train_data, test_data, user_num, item_num, train_mat = load_all()
#     train_dataset = NCFData(train_data, item_num, train_mat, num_ng, True)
#     train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
#     GMF_model, MLP_model = None, None
#     nn_model = NCF(user_num, item_num, factor_num, num_layers, dropout, config_model, GMF_model, MLP_model)
#     nn_model.cuda()
#     loss_function = nn.BCEWithLogitsLoss()
#     optimizer = optim.Adam(nn_model.parameters(), lr=lr)
#
#     for epoch in range(epochs):
#         nn_model.train()  # Enable dropout (if have).
#         train_loader.dataset.ng_sample()
#         #         print("Training starts...")
#         # iteration = 0
#         for user, item, label in train_loader:
#             user = user.cuda()
#             item = item.cuda()
#             label = label.float().cuda()
#
#             nn_model.zero_grad()
#             prediction = nn_model(user, item)
#             label = label.float()
#             loss = loss_function(prediction, label)
#             loss.backward()
#             optimizer.step()
#
#     nn_model.eval()
#
#     MAP = 0
#     nb_users = len(users_for_validate)
#     for user in users_for_validate:
#         top_n_list = recommend_topN_dnn(user, nn_model)
#         ap_user = average_precision(user, top_n_list)
#         MAP += ap_user
#     MAP = MAP / nb_users
#
#     return MAP
#
#
# tuning_params_dnn = {
#       "batch_size": (16, 512),
#       "factor_num": (8, 128),
#       "num_layers": (1, 8),
#       "dropout": (0, 1),
#       "lr": (0, 0.1),
#       "epochs": (5, 100),
#       "num_ng": (0, 10),
# }
#
# optimizer_dnn = BayesianOptimization(
#     f=optim_func_dnn,
#     pbounds=tuning_params_dnn,
#     verbose=0, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
#     random_state=1,
# )
#
# logger = JSONLogger(path="DNN/logs_dnn_PO.json")
# optimizer_dnn.subscribe(Events.OPTIMIZATION_STEP, logger)
#
#
# optimizer_dnn.maximize(
#     init_points=0,
#     n_iter=N_ITERATION_BO,
# )
