# Run M2M100-1.2B model on sample text. One GPU, default precision.

cd ..

python3 translate.py \
--sentences_path sample_text/en.txt \
--output_path sample_text/en2es.translation.m2m100_1.2B.txt \
--source_lang en \
--target_lang es \
--model_name facebook/m2m100_1.2B