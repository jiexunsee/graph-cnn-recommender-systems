from main import run
from itertools import product
import pickle

params_to_optimise = ['epochs', 'DO', 'hidden', 'layers'] # must correspond to below
iterations = 5

# NB_EPOCH = [125, 200]
# DO = [0.3, 0.5, 0.7]
# HIDDEN = [[100, 75], [300, 75]]
# NUM_LAYERS = [1, 2, 3]

# NB_EPOCH = [300, 400]
# DO = [0.7]
# HIDDEN = [[100, 75], [300, 75]]
# NUM_LAYERS = [2, 3, 4, 5]

NB_EPOCH = [300]
DO = [0.5, 0.7]
HIDDEN = [[75, 50], [150, 75], [200, 100]]
NUM_LAYERS = [1, 2, 3]

DATASET = 'flixster'
TESTING = False

for combo in product(NB_EPOCH, DO, HIDDEN, NUM_LAYERS):
	save_name = '-'.join(['{}_{}'.format(c, p) for p, c in zip(params_to_optimise, combo)])
	all_train_rmses, all_val_rmses, all_train_losses, all_val_losses, all_rmse = [], [], [], [], []
	for i in range(iterations):
		print('runnning experiment for {}'.format(save_name))
		params_dict = {'NB_EPOCH': combo[0], 'DO': combo[1], 'HIDDEN': combo[2], 'NUM_LAYERS': combo[3]}
		train_rmses, val_rmses, train_losses, val_losses, rmse = run(**params_dict)
		all_train_rmses.append(train_rmses)
		all_val_rmses.append(val_rmses)
		all_train_losses.append(train_losses)
		all_val_losses.append(val_losses)
		all_rmse.append(rmse)

	# if TESTING:
	# 	results = {'train_rmses': train_rmses, 'val_rmses': val_rmses, 'train_losses': train_losses, 'val_losses': val_losses, 'test_rmse': rmse}
	# else:
	# 	results = {'train_rmses': train_rmses, 'val_rmses': val_rmses, 'train_losses': train_losses, 'val_losses': val_losses, 'val_rmse': rmse}
	if TESTING:
		results = {'train_rmses': all_train_rmses, 'val_rmses': all_val_rmses, 'train_losses': all_train_losses, 'val_losses': all_val_losses, 'test_rmse': all_rmse}
	else:
		results = {'train_rmses': all_train_rmses, 'val_rmses': all_val_rmses, 'train_losses': all_train_losses, 'val_losses': all_val_losses, 'val_rmse': all_rmse}
	
	pickle.dump(results, open('results/'+save_name+'.pkl', 'wb'))
	print('Saved results at {}.pkl'.format(save_name))

print('hyperparameter search done!')