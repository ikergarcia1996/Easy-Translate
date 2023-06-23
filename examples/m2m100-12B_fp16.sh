# Run M2M100-1.2B model on sample text. We use FP16 precision, which requires a GPU with a lot of VRAM (i.e NVIDIA A100)
# For running this model in customer grade GPUs, use 8-bit quantization, see examples/m2m100-12B_8bit.sh

python3 translate.py \
--sentences_path sample_text/en.txt \
--output_path sample_text/en2es.translation.m2m100_12B.txt \
--source_lang en \
--target_lang es \
--model_name facebook/m2m100-12B-avg-5-ckpt \
--precision fp16