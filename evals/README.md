# Code for the evaluations presented in the paper
### SAEBench Benchmarks
* `absorb.py` uses [SAEBench](https://github.com/adamkarvonen/SAEBench) to generate absorption benchmark data
* `core.py` does the same for the downstream CE Loss/explained variance data
* `get_trained.py` is a simple utility for loading model weights
### Custom Translation Benchmark
* `english.json` contains the English sequences used, `translations.json` contains the translated sequences used
* `get_sequences.py` generates `english.json`
* `gemini_translate.py` translates the English sequences in `english.json`
* `run_models.py` runs the benchmark on the data, generating `translation_results.pkl`
* `process_bench_data.py` processes the translation benchmark results into numbers
* `get_model.py` is a simple utility for loading model weights
