# Trains Sparse Autoencoders based on activations from language models

All the main code to train and analyse Sparse Autoencoders is contained in the `sae` folder
1. `train.py` contains the training loop and defines the loss functions
2. `sparse_autoencoder.py` defines the `SparseAutoencoder` class and contains resampling functionality
3. `activation_store.py` defines the `ActivationStore` class that generates activations for a given model and dataset
4. `metrics.py` 

The 'sae_training_templates' folder contains example notebooks to get you started on training SAEs on open-source language models using different assumptions.

This package supports

1. Training on the residual stream, MLPs, attention head outputs, or concatenated attention head outputs.
2. Training an SAE where the input and output are different activations (sometimes referred to as transcoders).
3. Can be trained on any open-source hugging-face dataset or your own dataset for fine-tuning.
4. Basic SAE architecture can be modified in a variety of ways.
5. Gated SAEs, top-K SAEs, L0-based loss function, standard L1 loss function and Anthropic's L1 loss function.
6. Resampling of dead neurons
7. Training multiple SAEs in parallel
8. A variety of loss function customisations for avoiding dead features
9. Warmup of the L1/L0 coefficient

