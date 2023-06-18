
<p align="center">
    <br>
    <img src="images/title.png" width="900"/>
    <br>
<a href="https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2Fikergarcia1996%2FEasy-Translate"><img alt="Twitter" src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fikergarcia1996%2FEasy-Translate"></a>
<a href="https://github.com/ikergarcia1996/Easy-Translate/blob/main/LICENSE.md"><img alt="License" src="https://img.shields.io/github/license/ikergarcia1996/Easy-Translate"></a>
<a href="https://huggingface.co/docs/transformers/index"><img alt="Transformers" src="https://img.shields.io/badge/-%F0%9F%A4%97Transformers%20-grey"></a>
<a href="https://huggingface.co/docs/accelerate/index/"><img alt="Accelerate" src="https://img.shields.io/badge/-%F0%9F%A4%97Accelerate%20-grey"></a>
<a href="https://ikergarcia1996.github.io/Iker-Garcia-Ferrero/"><img alt="Author" src="https://img.shields.io/badge/Author-Iker García Ferrero-ff69b4"></a>

<br>
    <br>
</p>

Easy-Translate is a script for translating large text files with a 💥SINGLE COMMAND💥. Easy-Translate is designed to be as easy as possible for **beginners** and as **seamless** and **customizable** as possible for advanced users. 
We support almost any model, including [M2M100](https://arxiv.org/pdf/2010.11125.pdf),
[NLLB200](https://research.facebook.com/publications/no-language-left-behind/), 
[LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/),
[Bloom](https://bigscience.notion.site/BLOOM-BigScience-176B-Model-ad073ca07cdf479398d5f95d88e218c4) and more 🥳. 
We also provide a [script](#evaluate-translations) for Easy-Evaluation of your translations 📋

Easy-Translate is built on top of 🤗HuggingFace's [Transformers](https://huggingface.co/docs/transformers/index) and 🤗HuggingFace's [Accelerate](https://huggingface.co/docs/accelerate/index) library.


We currently support:

- CPU / multi-CPU / GPU / multi-GPU / TPU acceleration
- BF16 / FP16 / FP32 / 8 Bits / 4 Bits precision.
- Automatic batch size finder: Forget CUDA OOM errors. Set an initial batch size, if it doesn't fit, we will automatically adjust it.
- Multiple decoding strategies: Greedy Search, Beam Search, Top-K Sampling, Top-p (nucleus) sampling, etc. See [Decoding Strategies](#decodingsampling-strategies) for more information.
- :new: Load huge models in a single GPU with 8-bits / 4-bits quantization and support for splitting the model between GPU and CPU. See [Loading Huge Models](#loading-huge-models) for more information.
- :new: LoRA models support 
- :new: Support for any Seq2SeqLM or CausalLM model from HuggingFace's Hub.
- :new: Prompt support! See [Prompting](#prompting) for more information.

>Test the 🔌 Online Demo here: <https://huggingface.co/spaces/Iker/Translate-100-languages>



## Supported languages

See the [Supported languages table](supported_languages.md) for a table of the supported languages and their ids.

## Supported Models

💥 EasyTranslate now supports any Seq2SeqLM (m2m100, nllb200, MarianMT, T5, FlanT5, etc.) and any CausalLM (GPT2, LLaMA, Vicuna, Falcon) model from HuggingFace's Hub!!
We still recommend you to use M2M100 or NLLB200 for the best results, but you can experiment with other LLMs and prompting to generate translations. See [Prompting Section](#prompting) for more information. 

### M2M100
**M2M100** is a multilingual encoder-decoder (seq-to-seq) model trained for Many-to-Many multilingual translation introduced in this [paper](https://arxiv.org/abs/2010.11125) and first released in [this](https://github.com/pytorch/fairseq/tree/master/examples/m2m_100) repository. 
>M2M100 can directly translate between 9,900 directions of 100 languages.

- **Facebook/m2m100_418M**: <https://huggingface.co/facebook/m2m100_418M>

- **Facebook/m2m100_1.2B**: <https://huggingface.co/facebook/m2m100_1.2B>

- **Facebook/m2m100_12B**: <https://huggingface.co/facebook/m2m100-12B-avg-5-ckpt>

### NLLB200

**No Language Left Behind (NLLB)** open-sources models capable of delivering high-quality translations directly between any pair of 200+ languages — including low-resource languages like Asturian, Luganda, Urdu and more. It aims to help people communicate with anyone, anywhere, regardless of their language preferences. It was introduced in this [paper](https://research.facebook.com/publications/no-language-left-behind/) and first released in [this](https://github.com/facebookresearch/fairseq/tree/nllb) repository.
>NLLB can directly translate between +40,000 of +200 languages.

- **facebook/nllb-moe-54b**: <https://huggingface.co/facebook/nllb-moe-54b> (Requires transformers 4.28.0)

- **facebook/nllb-200-3.3B**: <https://huggingface.co/facebook/nllb-200-3.3B>

- **facebook/nllb-200-1.3B**: <https://huggingface.co/facebook/nllb-200-1.3B>

- **facebook/nllb-200-distilled-1.3B**: <https://huggingface.co/facebook/nllb-200-distilled-1.3B>

- **facebook/nllb-200-distilled-600M**: <https://huggingface.co/facebook/nllb-200-distilled-600M>


## Citation
If you use this software please cite
````
@inproceedings{garcia-ferrero-etal-2022-model,
    title = "Model and Data Transfer for Cross-Lingual Sequence Labelling in Zero-Resource Settings",
    author = "Garc{\'\i}a-Ferrero, Iker  and
      Agerri, Rodrigo  and
      Rigau, German",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.478",
    pages = "6403--6416",
}
````

## Requirements

```
Pytorch >= 1.10.0 
See: https://pytorch.org/get-started/locally/

Accelerate >= 0.12.0
pip install accelerate

HuggingFace Transformers 
If you plan to use NLLB200, please use >= 4.28.0, as an important bug was fixed in this version. 
pip install --upgrade transformers

BitsAndBytes (Optional, for 8-bits / 4bits quantization)
pip install bitsandbytes

PEFT (Optional, for LoRA models)
pip install peft
```

## Translate a file

Run `python translate.py -h` for more info.

#### Using a single CPU / GPU

```bash
python3 translate.py \
--sentences_path sample_text/en.txt \
--output_path sample_text/en2es.translation.m2m100_1.2B.txt \
--source_lang en \
--target_lang es \
--model_name facebook/m2m100_1.2B
```

#### Multi-GPU

See Accelerate documentation for more information (multi-node, TPU, Sharded model...): <https://huggingface.co/docs/accelerate/index>  
You can use the Accelerate CLI to configure the Accelerate environment (Run `accelerate config` in your terminal) instead of using the `--multi_gpu and --num_processes` flags.

```bash
# Use 2 GPUs
accelerate launch --multi_gpu --num_processes 2 --num_machines 1 translate.py \
--sentences_path sample_text/en.txt \
--output_path sample_text/en2es.translation.m2m100_1.2B.txt \
--source_lang en \
--target_lang es \
--model_name facebook/m2m100_1.2B
```


#### Automatic batch size finder

We will automatically find a batch size that fits in your GPU memory. The default initial batch size is 128 (You can set it with the `--starting_batch_size 128` flag). If we find an Out Of Memory error, we will automatically decrease the batch size until we find a working one.

### Loading Huge Models

Huge models such as LLaMA 65B or nllb-moe-54b can be loaded in a single GPU with 8 bits and 4 bits quantification with minimal performance degradation. 
See [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes). Set precision to 8 or 4 with the `--precision` flag. 

```bash
pip install bitsandbytes

python3 translate.py \
--sentences_path sample_text/en.txt \
--output_path sample_text/en2es.translation.nllb-moe-54b.txt \
--source_lang eng_Latn \
--target_lang spa_Latn \
--model_name facebook/nllb-moe-54b \
--precision 8 
```

If even the quantified model does not fit in your GPU memory, you can set the `--force_auto_device_map` flag. 
The model will be split across GPUs and CPU to fit it in memory. CPU offloading is slow, but will allow you to run huge models that do not fit in your GPU memory.



### Prompting

You can use LLMs such as LLaMA, Vicuna, GPT2, FlanT5, etc, instead of a translation model. These models require 
a prompt to define the task. You can either have the prompt already in the input file (each sentence includes the prompt) 
or you can use the `--prompt` flag to add the prompt to each sentence. In this case, you need to include the token %%SENTENCE%% in the prompt. 
This token will be replaced by the sentence to translate. You do not need to specify the `--source_lang` and `--target_lang` flags in this case.

```bash
python3 translate.py \
--sentences_path sample_text/en.txt \
--output_path sample_text/en2es.FlanT5.translation.txt \
--model_name google/flan-t5-large \
--prompt "Translate English to Spanish: %%SENTENCE%%" 
``` 


### Decoding/Sampling strategies

You can choose the decoding/sampling strategy to use and the number of candidate translations to output for each input sentence. 
By default, we will use beam-search with `num_beams` set to 5, and we will output the most likely candidate translation. This should be the best 
configuration for most use cases. You can change this behaviour with the following flags:

```bash
--num_beams: Number of beams to use for beam-search decoding (default: 5)
--do_sample: Whether to use sampling instead of beam-search decoding (default: False)
--temperature: Sampling temperature (default: 0.8)
--top_k: Top k sampling (default: 100)
--top_p: Top p sampling (default: 0.75)
--repetition_penalty: Repetition penalty (default: 1.0)
--keep_special_tokens: Whether to keep special tokens (default: False)
--keep_tokenization_spaces: Whether to keep tokenization spaces (default: False)
--num_return_sequences: Number of candidate translations to output for each input sentence (default: 1)
```
Please, note that running `--do_sample` with `--num_beams` > 1 and `8 bits` or `4 bits` quantification may be numerically unstable and produce an error. 

## Evaluate translations

To run the evaluation script you need to install [bert_score](https://github.com/Tiiiger/bert_score): `pip install bert_score` and 🤗HuggingFace's [Evaluate](https://huggingface.co/docs/evaluate) model: `pip install evaluate`.

The evaluation script will calculate the following metrics:

- [SacreBLEU](https://github.com/huggingface/datasets/tree/master/metrics/sacrebleu)
- [BLEU](https://github.com/huggingface/datasets/tree/master/metrics/bleu)
- [ROUGE](https://github.com/huggingface/datasets/tree/master/metrics/rouge)
- [METEOR](https://github.com/huggingface/datasets/tree/master/metrics/meteor)
- [TER](https://github.com/huggingface/datasets/tree/master/metrics/ter)
- [BertScore](https://github.com/huggingface/datasets/tree/master/metrics/bertscore)

Run the following command to evaluate the translations:

```bash
python3 eval.py \
--pred_path sample_text/en2es.translation.m2m100_1.2B.txt \
--gold_path sample_text/es.txt 
```

If you want to save the results to a file use the `--output_path` flag.

See [sample_text/en2es.m2m100_1.2B.json](sample_text/en2es.m2m100_1.2B.json) for a sample output.

