## 3 Sep 2018
**Progress:**
* Read the four papers listed in 16 Aug update
* Project plan completed

**Some insights:**
* Multigraphs: Graphs that are permitted to have multiple/parallel edges; where two vertices can be connected by more than one edge. However multi-graph in [Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks (Monti, 2017)](https://arxiv.org/abs/1704.06803) probably refers to how spectral convolution is applied to row and column graphs instead of just a single graph.

**Thoughts on applying residual gated graph convnets to recommender systems:**
* Building on encoder-decoder framework in GCMC:
  * Use RGGCN layer instead of message passing method in the encoder network
  * Use RGGCN layer on user and item graphs to produce additional feature information for 
* Building on framework in separable MGCNN:
  * Use RGCNN layer to produce **W** and **H** ?

**Some questions:**
* Sanity check: what does fixed/variable length graph mean? That number of vertices in a graph can change over time?
* RGGCN is applicable for graphs with variable length. GCMC looks like it can handle variable length? But not MGCNN?
* What does the test set look like for a recommender system matrix completion problem?


## 16 Aug 2018
**Brief outline of project:**

Investigate different graph CNN approaches as recommender systems. The 3 methods to be compared are as follows.
1. [Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks (Monti, 2017)](https://arxiv.org/abs/1704.06803) (MGCNN)
1. [Graph Convolutional Matrix Completion (Berg, 2017)](https://arxiv.org/abs/1706.02263) (GCMC)
1. [Residual Gated Graph ConvNets (Bresson, 2018)](https://arxiv.org/abs/1711.07553) (RGGCN)

[Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering (Defferrard, 2017)](https://arxiv.org/pdf/1606.09375.pdf) is an earlier paper that provides some theoretical background to the matrix completion method used in MGCNN by Monti et. al., 2017. 

The third approach (Bresson, 2018) proposes a novel architecture for graph learning tasks, incorporating gated edges and residuality with graph ConvNets. The challenge will be to apply this architecture to recommender system tasks.
