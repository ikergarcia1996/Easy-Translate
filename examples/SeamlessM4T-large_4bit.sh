# Runseamless-m4t-large model on sample text. This model requires a GPU with a lot of VRAM, so we use
# 8-bit quantization to reduce the required VRAM so we can fit in customer grade GPUs. If you have a GPU
# with a lot of RAM, running the model in FP16 should be faster and produce sighly better results,
# see examples/SeamlessM4T-large_bf16.sh

python3 translate.py \
--sentences_path sample_text/en.txt \
--output_path sample_text/en2es.translation.seamless-m4t-large.txt \
--source_lang eng \
--target_lang spa \
--model_name facebook/hf-seamless-m4t-large \
--precision 4 \
--starting_batch_size 8
