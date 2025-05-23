# Code for visualizing H-SAE features
* `16k_16_hsae.html` contains hundreds of features from the 16k x 16 H-SAE model
* `pregenerate_data.py` pre-generates a large dataset of H-SAE activations for visualization
* `SAEDashboard` is a heavily modified version of the [SAEDashboard](https://github.com/jbloomAus/SAEDashboard) library that must be installed with `uv pip install -e .` before proceeding
* `sae_dashboard_adapter.py` and `jax_sae_interface.py` are wrappers/monkeypatches for the SAEDashboard to allow interfacing with H-SAE models in JAX
* `Visualize Features.ipynb` is a notebook that takes some pre-generated activations and visualizes the features
* `get_model.py` is a simple utility for loading models weights and `get_logits.py` gets the unembed matrix and/or returns the output logits for reconstructed inputs
