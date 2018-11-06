## 12 Nov 2018
**Progress:**
* Experimented with simple feed forward layers as user and item feature extractors - results are the best so far though still far off those of GCMC

**To do:**
* Find out about edge gating from Prof
* Experiment with collaborative filtering in view of using its results as a baseline

## 29 Oct 2018
**Progress:**
* Tested results using amended code for [spatial graph convnets](https://github.com/xbresson/spatial_graph_convnets)
* Experimented with using more RGGCN layers in the GCMC recommender system encoder-decoder framework
* Using the same technique as [spatial graph convnets](https://github.com/xbresson/spatial_graph_convnets) of decaying the learning rate when training loss does not improve, a better test RMSE is obtained.

**Obstacles:**
* Tried to test GCMC with more layers but code fails at `sparse_tensor` operations for `support` and `support_t`. GCMC paper only uses one message passing layer.
* Experimenting with more RGGCN layers but still overfitting with poorer test accuracy than state-of-the-art.

**Thoughts:**
* Could a different architecture work better? E.g. edge gates are a major component of the RGGCN architecture but they may not be necessary in a recommender system task - all edges should probably be important in a recommender systems
* Is it better to predict a continuous value for ratings, rather than a softmax probability over the discrete rating types? Although having a softmax probability will allow the model to reflect that a user might either love or hate some item.

## 15 Oct 2018
**Progress:**
* Set up Google Cloud Platform GPU instance, installing CUDA 9.0 and CUDNN so that TensorFlow 1.10 runs with GPU acceleration.
* Translated PyTorch code from spatial graph convnets repo to TensorFlow code so that it runs within the framework by Berg et al. in GCMC
* Initial results [here](https://drive.google.com/drive/u/1/folders/1iid714S1XqTvL129w2zl_wYioqdR7sno). The models where RGGCN is used instead of the original message passing method in GCMC are severely overfitting. Train RMSE is very much lower than test RMSE. However, the train RMSE is lower than that of the original model for most datasets.
* Number of parameters much more using RGGCN. 16052760 parameters compared to 3048760 using GCMC. (more than 5x more)

**Questions:**
* Possible typo in [spatial graph convnets code](https://github.com/xbresson/spatial_graph_convnets/blob/master/01_residual_gated_graph_convnets_subgraph_matching.ipynb)
  * In line `x = Uix + torch.mm(E_end.t(), x1*x2) + self.bu1`, should it be `Ujx` instead of `Uix`?

## 17 Sep 2018
**Progress:**
* Read and understood code (TO DO)
  * [MGCNN code](https://github.com/fmonti/mgcnn)
  * [GCMC code](https://github.com/riannevdberg/gc-mc)
  * [RGGCN code](https://github.com/xbresson/spatial_graph_convnets)
* Ran code for MGCNN and GCMC on local machine
  * Reproduction results here: [google doc link](https://docs.google.com/document/d/1nU2W1fV3GRLtKmrvvLsA_G1Miu7KFXu7P2NjN1BoqXg/edit?usp=sharing)
* Preliminary modifying of code to replace message passing encoder step with RGGCN (TO DO)

**Questions:**
* MGCNN: why is parameters 1 for user+item graphs (Table 1) and m for user graph only (Table 2)?
* MGCNN code: how to tell if model is using user or user+item graphs?
* MGCNN: any reason why RMSE is higher when user+item graphs are used compared to when user graph is used for Flixster dataset?
* MGCNN code: movielens model runs out of CUDA memory at about 1120 iterations
* GCMC code: what's `polyak RMSE`?

**Observations:**

Both MGCNN and GCMC were trained on my personal computer with its GTX 1050 GPU. GCMC trains much faster than MGCNN. For example, GCMC takes about a minute to train on the Flixster dataset while MGCNN takes over an hour.

## 3 Sep 2018
**Progress:**
* Read the four papers listed in 16 Aug update
* Project plan completed

**Some insights:**
* Multigraphs: Graphs that are permitted to have multiple/parallel edges; where two vertices can be connected by more than one edge. However multi-graph in [Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks (Monti, 2017)](https://arxiv.org/abs/1704.06803) probably refers to how spectral convolution is applied to row and column graphs instead of just a single graph.

**Thoughts on applying residual gated graph convnets to recommender systems:**
* Building on encoder-decoder framework in GCMC:
  * Use RGGCN layer instead of message passing method in the encoder network to produce node embeddings
  * Use RGGCN layer on user and item graphs to produce additional feature information for encoder-decoder network
* Building on framework in MGCNN:
  * Use RGCNN layer to produce **W** and **H** as in a separable MGCNN?

**Some questions:**
* Sanity check: what does fixed/variable length graph mean? That number of vertices in a graph can change over time?
* RGGCN is applicable for graphs with variable length. GCMC looks like it can handle variable length? But not MGCNN?
* What does the test set look like for a recommender system matrix completion problem?


## 16 Aug 2018
**Brief outline of project:**

Investigate different graph CNN approaches as recommender systems. The 3 methods to be compared are as follows.
1. [Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks (Monti, 2017)](https://arxiv.org/abs/1704.06803) (Abbreviated as MGCNN in this project to avoid confusion with RGGCN)
1. [Graph Convolutional Matrix Completion (Berg, 2017)](https://arxiv.org/abs/1706.02263) (GCMC)
1. [Residual Gated Graph ConvNets (Bresson, 2018)](https://arxiv.org/abs/1711.07553) (RGGCN)

[Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering (Defferrard, 2017)](https://arxiv.org/pdf/1606.09375.pdf) is an earlier paper that provides some theoretical background to the matrix completion method used in MGCNN by Monti et. al., 2017. 

The third approach (Bresson, 2018) proposes a novel architecture for graph learning tasks, incorporating gated edges and residuality with graph ConvNets. The challenge will be to apply this architecture to recommender system tasks.
