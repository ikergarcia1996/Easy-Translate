# Run Mbart-many-to-many model on sample text.

cd ..

python3 translate.py \
--sentences_path sample_text/en.txt \
--output_path sample_text/en2es.translation.mbart.txt \
--source_lang en_XX \
--target_lang es_XX \
--model_name facebook/mbart-large-50-many-to-many-mmt