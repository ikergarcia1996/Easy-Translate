# Run M2M100-1.2B model on sample text. This model requires a GPU with a lot of VRAM, so we use
# 8-bit quantization to reduce the required VRAM so we can fit in customer grade GPUs. If you have a GPU
# with a lot of RAM, running the model in FP16 should be faster and produce sighly better results,
# see examples/m2m100-12B_fp16.sh

python3 translate.py \
--sentences_path sample_text/en.txt \
--output_path sample_text/en2es.translation.m2m100_12B.txt \
--source_lang en \
--target_lang es \
--model_name facebook/m2m100-12B-avg-5-ckpt \
--precision 8 \
--starting_batch_size 8