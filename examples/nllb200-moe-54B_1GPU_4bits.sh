# Run NLLB200-MOE model on sample text. This is a huge model that doesn't fit on a single GPU, so we use
# 4-bit quantization to reduce the required VRAM. Still it might not fit on a single GPU, so we also use
# the --force_auto_device_map flag that will offload the model parameters that don't fit on the GPU to the CPU.

python3 translate.py \
--sentences_path sample_text/en.txt \
--output_path sample_text/en2es.translation.nllb200-moe-54B.txt \
--source_lang eng_Latn \
--target_lang spa_Latn \
--model_name facebook/nllb-moe-54b \
--precision 4 \
--force_auto_device_map \
--starting_batch_size 8