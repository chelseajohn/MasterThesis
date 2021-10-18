# MasterThesis
Physics Master Thesis

In this thesis, we use the Hubbard Model to study the physics of correlated fermions in bipartite hexagonal and the special non-bipartite kagome lattice.To solve the Hubbard model we require to use stochastic methods like HMC since it is not always  possible to exactly diagonlize the system, as the underlying Hilbert space of n coupled spin half particles (dimension 2^{n}) exponentially increases with the size of the system.

The motivation behind the work done on the thesis comes from the success of using machine learning methods that are capable of learning highly non-linear functions.The thesis can be divided into three projects each of which, tries out a different regime of machine learning methods: Supervised Learning,Bayesian Learning,Unsupervised/Reinforcement Learning.

The first approach to introduce machine learning into HMC using supervised learning came from the work done by [Lingee Li](https://arxiv.org/abs/1711.05307).

The second approach of using Bayesian Learning was inspired by the success of the NNgHMC method in solving the Hubbard model.BNN are a unique combination of neural network and stochastic models with the stochastic model forming the core of this integration. BNNs can then produce probabilistic guarantees on it’s predictions and also generate the distribution of the parameters that it has learnt from the observations.That means, in the parameter space, one can deduce the nature and distribution of the neural network’s learnt parameters. The idea is to use these two characteristics of the BNN to estimate the error and confidence of the neural network model that approximates the force in leap frog steps for HMC. 

The third approach to use unsupervised/reinforcement learning to learn a MCMC kernel by training a modified Leapfrog integrator parametrized by neural nets was based on the paper [DanielLevy](https://arxiv.org/abs/1711.09268). 
