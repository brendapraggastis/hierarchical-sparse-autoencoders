# Hierarchical Sparse Autoencoder Repo
* This is the codebase for the Hierarchical Sparse Autoencoders paper!
* [sae](/sae) contains the implementation of the H-SAEs and code for running them on a small scale (only single GPU) using unembeddings
* [evals](/evals) contains code for running [SAEBench](https://github.com/adamkarvonen/SAEBench) absorption and core evals, alongside our custom translation evaluation
* [visualizations](/visualizations) contains HTML files to explore features in the H-SAE model and code for building more of these visualizations
* [build_training_data](/build_training_data) contains code for building the dataset of 1 billion embeddings from Wikipedia that were used to train the H-SAEs
* [download_weights](/download_weights) contains code for downloading our pre-trained SAEs
## Getting Started
* First, pull the repo with git and use [uv](https://docs.astral.sh/uv/) to install the necessary packages with `uv sync`
* If you'd like to just look at some features, go to [visualizations](/visualizations) and download the HTML file there
* To train an H-SAE on the Gemma unembeddings, there's a notebook in [sae](/sae) (runs easily on a single GPU!)
* Head to [download_weights](/download_weights) if you're intersted in running our pre-trained H-SAEs
* If you want to train your own, get started with [build_training_data](/build_training_data)
