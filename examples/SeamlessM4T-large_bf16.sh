# Run seamless-m4t-large model on sample text. We use FP16 precision, which requires a GPU with a lot of VRAM (i.e NVIDIA A100)
# For running this model in customer grade GPUs, use 4-bit quantization, see examples/SeamlessM4T-large_4bit.sh

python3 translate.py \
--sentences_path sample_text/en.txt \
--output_path sample_text/en2es.translation.seamless-m4t-large.txt \
--source_lang eng \
--target_lang spa \
--model_name facebook/hf-seamless-m4t-large \
--precision bf16 \
--starting_batch_size 8
