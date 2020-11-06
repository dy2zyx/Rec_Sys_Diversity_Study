import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import model
import config
import evaluate
import data_utils

# model folder
model_repo = 'models/'

# name options 'ml100k', 'ml1m', 'jester'
dataset_name = 'ml100k'

# model name
model_name = 'NeuMF-end_' + dataset_name + '.pth'

# load model
# neuMF = torch.load(model_repo + model_name)
# neuMF.eval()

# for testing
train_data, test_data, user_num ,item_num, train_mat = data_utils.load_all()

test_dataset = data_utils.NCFData(
		test_data, item_num, train_mat, 0, False)

test_loader = data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)

for user, item, label in test_loader:
	# print(user.item(), item.item(), label.item())

	gt_item = item[0].item()
	print(gt_item)

# top_k = 10
# HR, NDCG = evaluate.metrics(neuMF, test_loader, top_k)
# print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))