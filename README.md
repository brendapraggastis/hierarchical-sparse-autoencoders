# Hierarchical Sparse Autoencoder Repo
* This is the codebase for the Hierarchical Sparse Autoencoders paper!
* [sae](/sae) contains the implementation of the H-SAEs
* [evals](/evals) contains code for running [SAEBench](https://github.com/adamkarvonen/SAEBench) absorption and core evals, alongside our custom translation evaluation
* [visualizations](/visualizations) contains code for building interactive HTML pages to view H-SAE features
* [build_training_data](/build_training_data) contains code for building the dataset of 1 billion embeddings from Wikipedia that were used to train the H-SAEs
* [download_weights](/download_weights) contains code for downloading our pre-trained SAEs
## Getting Started
* First, pull the repo with git and use [uv](https://docs.astral.sh/uv/) to install the necessary packages with `uv sync`
* Then, head to [download_weights](/download_weights) if you're intersted in running our pre-trained H-SAEs
* If you want to train your own, get started with [build_training_data](/build_training_data)
