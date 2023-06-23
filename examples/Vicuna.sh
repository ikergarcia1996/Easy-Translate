# Run Vicuna1.3 model on sample text using prompting
# Different model sizes available, see https://github.com/lm-sys/FastChat#vicuna-weights:
# lmsys/vicuna-33b-v1.3
# lmsys/vicuna-13b-v1.3
# lmsys/vicuna-7b-v1.3

cd ..

python3 translate.py \
--sentences_path sample_text/en.txt \
--output_path sample_text/en2es.Vicuna33B.translation.txt \
--model_name lmsys/vicuna-33b-v1.3 \
--prompt "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: %%SENTENCE%% ASSISTANT:" \
--precision 8 \
--force_auto_device_map \
--starting_batch_size 8