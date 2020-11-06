import numpy as np
import torch
import heapq

from collections import defaultdict

def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def metrics(model, test_loader, top_k):
	HR, NDCG = [], []

	# for user, item, label in test_loader:
	# 	# user = user.cuda()
	# 	# item = item.cuda()
	# 	print(user, item, label)
	# 	predictions = model(user, item)
	# 	print(predictions)
	# 	_, indices = torch.topk(predictions, top_k)
	# 	recommends = torch.take(
	# 			item, indices).cpu().numpy().tolist()
	#
	# 	gt_item = item[0].item()
	# 	HR.append(hit(gt_item, recommends))
	# 	NDCG.append(ndcg(gt_item, recommends))

	user_predictions = defaultdict(list)
	for user, item, label in test_loader:
		# user = user.cuda()
		# item = item.cuda()
		prediction = model(user, item)

		user = user.detach().numpy().tolist()
		item = item.detach().numpy().tolist()
		prediction = prediction.detach().numpy().tolist()

		for index in range(len(user)):
			u = user[index]
			i = item[index]
			pred = prediction[index]
			user_predictions[u].append((i, pred))

	for user in user_predictions.keys():
		predictions = user_predictions[user]
		gt_item = predictions[0]
		# print(gt_item)
		recommends = heapq.nlargest(top_k, predictions, key=lambda x: x[1])
		HR.append(hit(gt_item, recommends))
		NDCG.append(ndcg(gt_item, recommends))
	return np.mean(HR), np.mean(NDCG)
