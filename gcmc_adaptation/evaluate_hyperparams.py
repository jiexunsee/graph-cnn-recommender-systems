from main import run
from itertools import product
import pickle
import numpy as np

iterations = 5

NB_EPOCH = 3000
DO = 0.5
HIDDEN = [100, 25]
NUM_LAYERS = 2

DATASET = 'facebook'
TESTING = True
VERBOSE = False
FEATURES = True
ACCUM = 'stack'
DROPOUT_EDGES = False
VF = 0
TF = 0.2
OF = 0.8

# all_train_rmses, all_val_rmses, all_train_losses, all_val_losses, all_rmse, all_accuracy, all_precision, all_recall, all_auc = [], [], [], [], [], [], [], [], []
# experiment_name = '{} with {}, DE {}'.format(DATASET, ACCUM, DROPOUT_EDGES)
# for i in range(iterations):
# 	print('runnning experiment for {}'.format(experiment_name))
# 	params_dict = {'DATASET': DATASET, 'FEATURES': FEATURES, 'NB_EPOCH': NB_EPOCH, 'DO': DO, 'HIDDEN': HIDDEN, 'NUM_LAYERS': NUM_LAYERS, 'TESTING': TESTING,
# 					'VERBOSE': VERBOSE, 'ACCUM': ACCUM, 'DROPOUT_EDGES': DROPOUT_EDGES, 'val_fraction': VF, 'test_fraction': TF, 'observed_fraction': OF}
# 	train_rmses, val_rmses, train_losses, val_losses, rmse, accuracy, precision, recall, auc = run(**params_dict)
# 	all_train_rmses.append(train_rmses)
# 	all_val_rmses.append(val_rmses)
# 	all_train_losses.append(train_losses)
# 	all_val_losses.append(val_losses)
# 	all_rmse.append(rmse)
# 	all_accuracy.append(accuracy)
# 	all_precision.append(precision)
# 	all_recall.append(recall)
# 	all_auc.append(auc)

# print('hyperparameter evaluation done! for {} test fraction]'.format(TF))

# best_epochs = np.argmin(np.array(all_val_rmses), 1)
# average = np.mean(all_rmse)
# print('best epochs: {}'.format(best_epochs))
# print('average rmse: {}'.format(average))
# print('average accuracy: {}'.format(np.mean(all_accuracy)))
# print('average precision: {}'.format(np.mean(all_precision)))
# print('average recall: {}'.format(np.mean(all_recall)))
# print('average auc: {}'.format(np.mean(all_auc)))
# print('rmse scores: {}'.format(all_rmse))
# print('accuracy scores: {}'.format(all_accuracy))
# print('precision scores: {}'.format(all_precision))
# print('recall scores: {}'.format(all_recall))
# print('auc scores: {}'.format(all_auc))



ACCUM = 'stackRGGCN'

all_train_rmses, all_val_rmses, all_train_losses, all_val_losses, all_rmse, all_accuracy, all_precision, all_recall, all_auc = [], [], [], [], [], [], [], [], []
experiment_name = '{} with {}, DE {}'.format(DATASET, ACCUM, DROPOUT_EDGES)
for i in range(iterations):
	print('runnning experiment for {}'.format(experiment_name))
	params_dict = {'DATASET': DATASET, 'FEATURES': FEATURES, 'NB_EPOCH': NB_EPOCH, 'DO': DO, 'HIDDEN': HIDDEN, 'NUM_LAYERS': NUM_LAYERS, 'TESTING': TESTING,
					'VERBOSE': VERBOSE, 'ACCUM': ACCUM, 'DROPOUT_EDGES': DROPOUT_EDGES, 'val_fraction': VF, 'test_fraction': TF, 'observed_fraction': OF}
	train_rmses, val_rmses, train_losses, val_losses, rmse, accuracy, precision, recall, auc = run(**params_dict)
	all_train_rmses.append(train_rmses)
	all_val_rmses.append(val_rmses)
	all_train_losses.append(train_losses)
	all_val_losses.append(val_losses)
	all_rmse.append(rmse)
	all_accuracy.append(accuracy)
	all_precision.append(precision)
	all_recall.append(recall)
	all_auc.append(auc)

print('hyperparameter evaluation done! for {} test fraction]'.format(TF))

best_epochs = np.argmin(np.array(all_val_rmses), 1)
average = np.mean(all_rmse)
print('FOR RGGCN:')
print('best epochs: {}'.format(best_epochs))
print('average rmse: {}'.format(average))
print('average accuracy: {}'.format(np.mean(all_accuracy)))
print('average precision: {}'.format(np.mean(all_precision)))
print('average recall: {}'.format(np.mean(all_recall)))
print('average auc: {}'.format(np.mean(all_auc)))
print('rmse scores: {}'.format(all_rmse))
print('accuracy scores: {}'.format(all_accuracy))
print('precision scores: {}'.format(all_precision))
print('recall scores: {}'.format(all_recall))
print('auc scores: {}'.format(all_auc))