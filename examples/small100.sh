# Run SMALL100 model on sample text.

python3 translate.py \
--sentences_path sample_text/en.txt \
--output_path sample_text/en2es.translation.small100.txt \
--source_lang en \
--target_lang es \
--model_name alirezamsh/small100