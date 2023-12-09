# Bayesian Flow Networks

A Bayesian Flow Network (bfn) is a generative model that produces an output distribution based on a set of independent input distributions that are optimized using Bayesian inference. For more details, refer to the [paper](https://arxiv.org/abs/2308.07037). This repository is a simple implementation of a continous data, discrete time bfn for educational purposes. Thus, comments in the bfn implementation 'bfn/src/bayesian_flow_network/bfn.py' refer to the paper.

## How BFNs Work in a Nutshell

A sample from a sender distribution (noisy data point) is used to update an input distribution which is then used to receive an output distribution from a neural network. 
<div align="left">
  <img src="assets/sro_dist.png" alt="SRO Distribution" width="35%">
</div>

Based on the observations (sender distribution samples), the input distributions posterior is continuously updated...
<div align="left">
  <img src="assets/update_dist.png" alt="Update Distribution" width="50%">
</div>

.. which creates an updated set of input distributions.
<div align="left">
  <img src="assets/updated_dist.png" alt="Updated Distribution" width="40%">
</div>

## Installation

```bash
pip install torch numpy matplotlib
```

# Test Case

The test case `bfn/src/test_sinusoidal.py` trains a Bayesian Flow Network (BFN) to generate sinusoidal curves with random amplitudes, frequencies, and phases.

<div align="left">
  <img src="assets/gen_sin.gif" alt="Generated Sinusoidal Data" width="40%">
</div>


