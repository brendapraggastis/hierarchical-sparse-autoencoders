# Building Training Data!
* This is the code to build the training data for training the H-SAE models
* `load_wikipedia.py` loads the 20231101 [Wikipedia dump](huggingface.co/datasets/wikimedia/wikipedia) from Huggingface into a postgres SQL database
* `add_codes.ipynb` loads the number of active users for each Wikipedia from `table.csv` and labels the SQL database with the language code for each Wikipedia
* `get_lengths.py` gets the number of tokens in each Wikipedia article and `analyze_lengths.ipynb` uses this added length data + the language code data to select which articles will be selected for the training data
* `generate_tokens.py` generates the tokens for each article and packs them together into Gemma context window sized chunks into parquet files
* `process_tokens.py` takes the parquet file and generates a [Tensorstore](https://google.github.io/tensorstore/index.html) dataset which is used by the rest of the H-SAE codebase as a "source of truth" for the tokens
* `build_attn_mask.py` builds an attention mask for the packed sequences of articles
* `generate_embeddings.py` takes the tokens tensorstore and makes a tensorstore of the embeddings/activations from Gemma
* `shuffle.py` takes the generated activations tensorstore dataset and shuffles + re-chunks it for faster reads during training (our training was on a 8xA100 and needs >1GB/s to saturate the node)
* **NOTE**: To build the full training set ~15TB of storage is needed
