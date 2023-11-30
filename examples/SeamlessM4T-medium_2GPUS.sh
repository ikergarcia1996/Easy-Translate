# Run  seamless-m4t-medium on sample text. Multi GPU, default precision.

accelerate launch --multi_gpu --num_processes 2 --num_machines 1 translate.py \
--sentences_path sample_text/en.txt \
--output_path sample_text/en2es.translation.seamless-m4t-medium.txt \
--source_lang eng \
--target_lang spa \
--model_name facebook/hf-seamless-m4t-medium
