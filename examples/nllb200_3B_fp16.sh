# Run NLLB200-3B model on sample text. We use FP16 precision, which requires a GPU with a lot of VRAM
# For running this model in GPUs with less VRAM, use 8-bit quantization, see examples/nllb200_3B_8bit.sh

cd ..

python3 translate.py \
--sentences_path sample_text/en.txt \
--output_path sample_text/en2es.translation.nllb-200_3B.txt \
--source_lang eng_Latn \
--target_lang spa_Latn \
--model_name facebook/nllb-200-3.3B \
--precision fp16