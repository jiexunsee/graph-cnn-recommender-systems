import numpy as np
from glob import glob


##### GATHERING SIDE INFO #####
def gather_side_info():
	featname_files = glob('*.featnames')
	egos = []
	for file in featname_files:
		ego = file.split('.')[0]
		egos.append(ego)
	egos.sort()

	all_featnames = []
	for ego in egos:
		file = '{}.featnames'.format(ego)
		names = open(file, 'r').read()
		names = names.split('\n')
		for name in names:
			if len(name) < 2:
				continue
			s = name.split(' ')
			name = (' ').join(s[1:])
			all_featnames.append(name)
	all_featnames
	all_featnames = list(set(all_featnames))
	all_featnames.sort()
	print('number of unique feature names: {}'.format(len(all_featnames))) # 1406 features. Similar to 3000 for flixster dataset.
	featnames_dict = {all_featnames[i]: i for i in range(len(all_featnames))}

	user_features = np.zeros((4039, len(all_featnames)))

	def process_ego_network(ego):
		feat = open('{}.feat'.format(ego), 'r').read()
		names = open('{}.featnames'.format(ego), 'r').read()
		names = names.split('\n')
		featnames = []
		for name in names:
			if len(name) < 2:
				continue
			s = name.split(' ')
			name = (' ').join(s[1:])
			featnames.append(name)
		for f in feat.split('\n'):
			if len(f) < 2:
				continue
			f = f.split()
			f = [int(a) for a in f]
			positives = np.where(f[1:])
			for p in positives[0]:
				positive_feat = featnames[p]
				positive_index = featnames_dict[positive_feat]
				user_features[f[0], positive_index] = 1

		file = '{}.egofeat'.format(ego)
		egofeat = open(file, 'r').read()
		egofeat = egofeat.split()
		egofeat = [int(a) for a in egofeat]
		positives = np.where(egofeat)
		for p in positives[0]:
			positive_feat = featnames[p]
			positive_index = featnames_dict[positive_feat]
			user_features[int(ego), positive_index] = 1

	for ego in egos:
		process_ego_network(ego)

	nonzero = len(np.where(user_features)[0])
	print('number of nonzero entries: {} out of {}. ({}%)'.format(nonzero, 4039*len(all_featnames), nonzero/(4039*len(all_featnames))))
	w = np.where(user_features[3437])[0]
	print(len(w))
	for i in w:
		print(all_featnames[i])

	np.save('user_features.npy', user_features)

def gather_edges_info():
	adj = np.zeros((4039, 4039))
	file = 'facebook_combined.txt'
	edges = open(file, 'r').read()
	edges = edges.split('\n')
	for edge in edges:
		if len(edge) < 2:
			continue
		s = int(edge.split()[0])
		e = int(edge.split()[1])
		adj[s, e] = 1
		adj[e, s] = 1
	np.save('adjacency1.npy', adj)
	return adj

if __name__ == '__main__':
	adj = gather_edges_info()
	print(adj)
	print(len(np.where(adj)[0])/2)