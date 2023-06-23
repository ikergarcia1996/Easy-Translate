# Run OpusMT model on sample text.

python3 translate.py \
--sentences_path sample_text/en.txt \
--output_path sample_text/en2es.translation.opusMT.txt \
--model_name Helsinki-NLP/opus-mt-en-es