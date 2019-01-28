from main import run
from itertools import product
import pickle
import numpy as np

params_to_optimise = ['epochs', 'DO', 'hidden', 'layers'] # must correspond to below
iterations = 5

NB_EPOCH = 300
DO = 0.5
HIDDEN = [75, 50]
NUM_LAYERS = 2

DATASET = 'flixster'
TESTING = False
VERBOSE = False
FEATURES = True
ACCUM = 'stackRGGCN'
DROPOUT_EDGES = True

all_train_rmses, all_val_rmses, all_train_losses, all_val_losses, all_rmse = [], [], [], [], []
for i in range(iterations):
	print('runnning experiment for {}'.format(save_name))
	params_dict = {'DATASET': DATASET, 'FEATURES': FEATURES, 'NB_EPOCH': combo[0], 'DO': combo[1], 'HIDDEN': combo[2], 'NUM_LAYERS': combo[3],
					'VERBOSE': VERBOSE, 'ACCUM': ACCUM, 'DROPOUT_EDGES': DROPOUT_EDGES}
	train_rmses, val_rmses, train_losses, val_losses, rmse = run(**params_dict)
	all_train_rmses.append(train_rmses)
	all_val_rmses.append(val_rmses)
	all_train_losses.append(train_losses)
	all_val_losses.append(val_losses)
	all_rmse.append(rmse)

print('hyperparameter evaluation done!')

best_epochs = np.argmin(np.atleast_2d(np.array(all_val_rmses), 1))
average = np.mean(all_rmse)
print('best epochs: {}'.format(best_epochs))
print('average val rmse: {}'.format(average))