# Bayesian Flow Networks

A Bayesian Flow Network (BFN) is a generative model that integrates Bayesian inference with neural networks. For more details, refer to the [paper](https://arxiv.org/abs/2308.07037).

## How BFNs Work in a Nutshell

BFNs transmit data from a sender to a receiver distribution and receive an output distribution from a neural network. 
<div align="left">
  <img src="assets/sro_dist.png" alt="SRO Distribution" width="35%">
</div>

Based on their observations, their prior is continuously updated...
<div align="left">
  <img src="assets/update_dist.png" alt="Update Distribution" width="50%">
</div>

.. which creates an updated distribution.
<div align="left">
  <img src="assets/updated_dist.png" alt="Updated Distribution" width="40%">
</div>


# Test Case

The test case `bfn/src/test_sinosoidal.py` trains a Bayesian Flow Network (BFN) to generate sinusoidal curves with random amplitudes, frequencies, and phases.

<div align="left">
  <img src="assets/gen_sin.gif" alt="Generated Sinusoidal Data" width="40%">
</div>
