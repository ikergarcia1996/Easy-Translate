# Run OpusMT model on sample text.

cd ..

python3 translate.py \
--sentences_path sample_text/en.txt \
--output_path sample_text/en2es.translation.opus.txt \
--model_name Helsinki-NLP/opus-mt-es-en