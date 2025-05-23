# H-SAE Implementation code

### Implementation Files
* `moe_eqx.py` contains the model itself
* `run_moe_eqx.py` orchestrates the training job/checkpointing
* `run_moe_eqx_utils.py` contains utility functions for training/running (e.g. a function to load a checkpointed model)
### Unembeddings Files
* `clean_gemma_embeddings.ipynb` creates the whitened and cleaned Gemma vocabulary
* `download_gemma_embeddings.py` alternately simply downloads them from S3
* `run_hsae_unembeddings.ipynb` shows how to run the model on the unembeddings
