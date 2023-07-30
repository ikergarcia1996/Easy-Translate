# Run  NLLB200-3B on sample text. This model requires a GPU with a lot of VRAM, so we use
# 8-bit quantization to reduce the required VRAM so we can fit in customer grade GPUs. If you have a GPU
# with a lot of RAM, running the model in FP16 should be faster and produce sighly better results,
# see examples/nllb200-3B_fp16.sh


python3 translate.py \
--sentences_path sample_text/en.txt \
--output_path sample_text/en2es.translation.nllb-200_3B.txt \
--source_lang eng_Latn \
--target_lang spa_Latn \
--model_name facebook/nllb-200-3.3B \
--precision 8