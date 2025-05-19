# TEST / EXAMPLE
neuronpedia-runner \
  --sae-set="gemma-scope-2b-pt-res-canonical" \
  --sae-path="layer_0/width_16k/canonical" \
  --np-set-name="gemmascope-res-16k" \
  --np-sae-id-suffix="l0_39" \
  --dataset-path="monology/pile-uncopyrighted" \
  --output-dir="neuronpedia_outputs/" \
  --sae_dtype="float32" \
  --model_dtype="bfloat16" \
  --sparsity-threshold=1 \
  --n-prompts=128 \
  --n-tokens-in-prompt=128 \
  --n-prompts-in-forward-pass=128 \
  --n-features-per-batch=2 \
  --start-batch=0 \
  --end-batch=6 \
  --use-wandb