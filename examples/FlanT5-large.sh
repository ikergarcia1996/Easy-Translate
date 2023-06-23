# Run Flan-T5 model on sample text using promting

python3 translate.py \
--sentences_path sample_text/en.txt \
--output_path sample_text/en2es.FlanT5.translation.txt \
--model_name google/flan-t5-large \
--prompt "Translate English to Spanish: %%SENTENCE%%" \
--precision bf16