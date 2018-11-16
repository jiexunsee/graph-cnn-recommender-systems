from main import run
from itertools import product
import pickle

params_to_optimise = ['epochs', 'DO', 'hidden', 'layers'] # must correspond to below

NB_EPOCH = [125, 200]
DO = [0.3, 0.5, 0.7]
HIDDEN = [[100, 75], [300, 75]]
NUM_LAYERS = [1, 2, 3]

# NB_EPOCH = [2, 2]
# DO = [0.7]
# HIDDEN = [[100, 75]]
# NUM_LAYERS = [3]

DATASET = 'douban'
TESTING = False

for combo in product(NB_EPOCH, DO, HIDDEN, NUM_LAYERS):
	save_name = '-'.join(['{}_{}'.format(c, p) for p, c in zip(params_to_optimise, combo)])
	print('runnning experiment for {}'.format(save_name))
	params_dict = {'NB_EPOCH': combo[0], 'DO': combo[1], 'HIDDEN': combo[2], 'NUM_LAYERS': combo[3]}
	train_rmses, val_rmses, train_losses, val_losses, rmse = run(**params_dict)

	if TESTING:
		results = {'train_rmses': train_rmses, 'val_rmses': val_rmses, 'train_losses': train_losses, 'val_losses': val_losses, 'test_rmse': rmse}
	else:
		results = {'train_rmses': train_rmses, 'val_rmses': val_rmses, 'train_losses': train_losses, 'val_losses': val_losses, 'val_rmse': rmse}
	pickle.dump(results, open('results/'+save_name+'.pkl', 'wb'))

print('hyperparameter search done!')