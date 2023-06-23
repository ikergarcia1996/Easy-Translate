# Run LLaMA65B model on sample text using prompting

python3 translate.py \
--sentences_path sample_text/en.txt \
--output_path sample_text/en2es.LLaMA.translation.txt \
--model_name PATH_TO_LOCAL_LLAMA_WEIGHTS_IN_HF_FORMAT \
--prompt "Translate English to Spanish: %%SENTENCE%%" \
--precision 8 \
--precision 4 \
--force_auto_device_map \
--starting_batch_size 8