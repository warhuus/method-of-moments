# method-of-moments

This is a numpy implementation of Algorithm B from "A Method of Moments for Mixture Models and Hidden Markov Models" by A. Anandkumar, D. Hsu, and S.M. Kakade. You can find the original paper here: https://arxiv.org/abs/1203.0683. The implementation is specifically for estimating the parameters of Hidden Markov Models (HMMs) with multidimensional Gaussian emmisions. 

The overall structure is taken from maxentile's "method-of-moments-tinker" repository (https://github.com/maxentile/method-of-moments-tinker). Much  inspiration has been taken from Carl Mattfeld's implementation for discrete emissions (https://github.com/cmgithub/spectral). Many thanks to both!

Requires numpy and scipy.
