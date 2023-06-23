# Run Alpaca-Lora (A LoRA model) model on sample text using prompting
# We need to set the base model with --model_name and the LoRA weights with --lora_weights_name_or_path

cd ..

python3 translate.py \
--sentences_path sample_text/en.txt \
--output_path sample_text/en2es.AlpacaLora.translation.txt \
--model_name decapoda-research/llama-7b-hf \
--lora_weights_name_or_path tloen/alpaca-lora-7b \
--prompt "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nTranslate this text from English into Spanish\n\n### Input:\n%%SENTENCE%%\n\n### Response:\n" \
--precision 8 \
--force_auto_device_map \
--starting_batch_size 8