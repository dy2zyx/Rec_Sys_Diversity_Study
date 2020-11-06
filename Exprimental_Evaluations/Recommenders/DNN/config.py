# dataset name 
dataset = 'ml1m'
# assert dataset in ['ml100k', 'ml1m', 'jester']
assert dataset in ['anime', 'ml1m', 'lbthing']

# model name 
model = 'NeuMF-end'
assert model in ['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre']

# paths
main_path = '/Users/yudu/PycharmProjects/PHD_Yu/DiversifyRec/datasets/' + dataset + '/'

train_rating = main_path + 'train.csv'
test_rating = main_path + 'test.csv'
test_negative = main_path + 'test.negative'

model_path = '/Users/yudu/PycharmProjects/PHD_Yu/DiversifyRec/neuCF_models/'
GMF_model_path = model_path + 'GMF.pth'
MLP_model_path = model_path + 'MLP.pth'
NeuMF_model_path = model_path + 'NeuMF.pth'
