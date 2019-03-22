## Experimental Comparison of Recommender Systems

### Project summary
The goal of the project is to evaluate several state-of-the-art recommender systems applied to benchmark datasets. Particular attention will be given to convolutional neural networks on graphs.

This is a Final Year Project by See Jie Xun, supervised by A/P Xavier Bresson. The project website can be found [here](https://jiexunsee.github.io/graph-cnn-recommender-systems).

### Packages required (Python 3)
* tensorflow >= 1.13 (If possible, use tensorflow-GPU for faster training of models)
* numpy >= 1.14
* scipy >= 1.1
* scikit-learn >= 0.20
* pandas >= 0.23
* h5py >= 2.7

### Data
The datasets are already contained in the repository, under the "gcmc_adaptation/data" folder. The Douban, Flixster, and YahooMusic datasets are taken from the [repository](https://github.com/riannevdberg/gc-mc) for the Graph Convolutional Matrix Completion method by van den Berg et al.

The Facebook dataset needs to be processed to generate an adjacency matrix and user features, for us to be able to train a GCN on it. To do this, run the `gcmc_adaptation/data/facebook/processing.py` script, which will save the adjacency matrix as `adjacency2.npy` in the same folder. The user features have already been generated and are saved as `user_features.npy`.

### Scripts to use
To run a single experiment, use the script `gcmc_adaptation/train.py`. Hyperparameter and experimental settings need to be specified at the command line when running the script. For example, use the command to run an experiment on the Facebook dataset:

```
python train.py -d facebook --accum stackRGGCN --dropout 0.5 -e 1000 --hidden 50 25 --num_layers 2 --features --testing
```
Note: The `--accum` flag selects the GCN architecture to be used to generate representations for each node based on the graph structure and node features. The primary focus of this project was on investigating the [RGGCN](https://github.com/xbresson/spatial_graph_convnets) architecture as applied to this graph autoencoder framework for recommender systems. This can be selected by specifying `--accum stackRGGCN`.

To evaluate the results of a set of hyperparameters, averaged over a number of iterations, use the script `gcmc_adaptation/evaluate_hyperparameters.py`. The hyperparameters and number of iterations can be edited within this script.
