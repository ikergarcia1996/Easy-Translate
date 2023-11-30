# Run seamless-m4t-medium model on sample text. One GPU, default precision.

python3 translate.py \
--sentences_path sample_text/en.txt \
--output_path sample_text/en2es.translation.seamless-m4t-medium.txt \
--source_lang eng \
--target_lang spa \
--model_name facebook/hf-seamless-m4t-medium
