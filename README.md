
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

Easy-Translate is a script for translating large text files in your machine using the [M2M100 models](https://arxiv.org/pdf/2010.11125.pdf) and [NLLB200 models](https://research.facebook.com/publications/no-language-left-behind/) from Facebook/Meta AI.  We also privide a [script](#evaluate-translations) for Easy-Evaluation of your translations 🥳

Easy-Translate is built on top of 🤗HuggingFace's [Transformers](https://huggingface.co/docs/transformers/index) and 🤗HuggingFace's [Accelerate](https://huggingface.co/docs/accelerate/index) library.

We currently support:

- CPU / multi-CPU / GPU / multi-GPU / TPU acceleration
- BF16 / FP16 / FP32 precision.
- Automatic batch size finder: Forget CUDA OOM errors. Set an initial batch size, if it doesn't fit, we will automatically adjust it.
- Sharded Data Parallel to load huge models sharded on multiple GPUs (See: <https://huggingface.co/docs/accelerate/fsdp>).
- Greedy decoding / Beam Search decoding / Multinomial Sampling / Beam-Search Multinomial Sampling

>Test the 🔌 Online Demo here: <https://huggingface.co/spaces/Iker/Translate-100-languages>



## Supported languages

See the [Supported languages table](supported_languages.md) for a table of the supported languages and their ids.

## Supported Models

### M2M100
**M2M100** is a multilingual encoder-decoder (seq-to-seq) model trained for Many-to-Many multilingual translation introduced in this [paper](https://arxiv.org/abs/2010.11125) and first released in [this](https://github.com/pytorch/fairseq/tree/master/examples/m2m_100) repository. 
>M2M100 can directly translate between 9,900 directions of 100 languages.

- **Facebook/m2m100_418M**: <https://huggingface.co/facebook/m2m100_418M>

- **Facebook/m2m100_1.2B**: <https://huggingface.co/facebook/m2m100_1.2B>

- **Facebook/m2m100_12B**: <https://huggingface.co/facebook/m2m100-12B-avg-5-ckpt>

### NLLB200

**No Language Left Behind (NLLB)** open-sources models capable of delivering high-quality translations directly between any pair of 200+ languages — including low-resource languages like Asturian, Luganda, Urdu and more. It aims to help people communicate with anyone, anywhere, regardless of their language preferences. It was introduced in this [paper](https://research.facebook.com/publications/no-language-left-behind/) and first released in [this](https://github.com/facebookresearch/fairseq/tree/nllb) repository.
>NLLB can directly translate between +40,000 of +200 languages.

- **facebook/nllb-200-3.3B**: <https://huggingface.co/facebook/nllb-200-3.3B>

- **facebook/nllb-200-1.3B**: <https://huggingface.co/facebook/nllb-200-1.3B>

- **facebook/nllb-200-distilled-1.3B**: <https://huggingface.co/facebook/nllb-200-distilled-1.3B>

- **facebook/nllb-200-distilled-600M**: <https://huggingface.co/facebook/nllb-200-distilled-600M>


Any other ModelForSeq2SeqLM from HuggingFace's Hub should work with this library: <https://huggingface.co/models?pipeline_tag=text2text-generation>

## Requirements

```
Pytorch >= 1.10.0
See: https://pytorch.org/get-started/locally/

Accelerate >= 0.12.0
pip install --upgrade accelerate

HuggingFace Transformers 
pip install --upgrade transformers

If you find errors using NLLB200, try installing transformers from source:
pip install git+https://github.com/huggingface/transformers.git
```

## Translate a file

Run `python translate.py -h` for more info.

#### Using a single CPU / GPU

```bash
accelerate launch translate.py \
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

#### Choose precision

Use the `--precision` flag to choose the precision of the model. You can choose between: bf16, fp16 and 32.

```bash
accelerate launch translate.py \
--sentences_path sample_text/en.txt \
--output_path sample_text/en2es.translation.m2m100_1.2B.txt \
--source_lang en \
--target_lang es \
--model_name facebook/m2m100_1.2B \
--precision fp16 
```

### Decoding/Sampling strategies

You can choose the decoding/sampling strategy to use and the number of candidate translation to output for each input sentence. By default we will use beam-search with 'num_beams' set to 5, and we will output the most likely candidate translation. But you can change this behavior:
##### Greedy decoding
```bash
accelerate launch translate.py \
--sentences_path sample_text/en.txt \
--output_path sample_text/en2es.translation.m2m100_1.2B.txt \
--source_lang en \
--target_lang es \
--model_name facebook/m2m100_1.2B \
--num_beams 1 
```

##### Multinomial Sampling 
```bash
accelerate launch translate.py \
--sentences_path sample_text/en.txt \
--output_path sample_text/en2es.translation.m2m100_1.2B.txt \
--source_lang en \
--target_lang es \
--model_name facebook/m2m100_1.2B \
--num_beams 1 \
--do_sample \
--temperature 0.5 \
--top_k 100 \
--top_p 0.8 \
--num_return_sequences 1
```
##### Beam-Search decoding **(DEFAULT)**
```bash
accelerate launch translate.py \
--sentences_path sample_text/en.txt \
--output_path sample_text/en2es.translation.m2m100_1.2B.txt \
--source_lang en \
--target_lang es \
--model_name facebook/m2m100_1.2B \
--num_beams 5 \
--num_return_sequences 1 \ 
```
##### Beam-Search Multinomial Sampling
```bash
accelerate launch translate.py \
--sentences_path sample_text/en.txt \
--output_path sample_text/en2es.translation.m2m100_1.2B.txt \
--source_lang en \
--target_lang es \
--model_name facebook/m2m100_1.2B \
--num_beams 5 \
--num_return_sequences 1 \
--do_sample \
--temperature 0.5 \
--top_k 100 \
--top_p 0.8 
```

## Evaluate translations

To run the evaluation script you need to install [bert_score](https://github.com/Tiiiger/bert_score): `pip install bert_score` and 🤗HuggingFace's [Datasets](https://huggingface.co/docs/datasets/index) model: `pip install datasets`.

The evaluation script will calculate the following metrics:

- [SacreBLEU](https://github.com/huggingface/datasets/tree/master/metrics/sacrebleu)
- [BLEU](https://github.com/huggingface/datasets/tree/master/metrics/bleu)
- [ROUGE](https://github.com/huggingface/datasets/tree/master/metrics/rouge)
- [METEOR](https://github.com/huggingface/datasets/tree/master/metrics/meteor)
- [TER](https://github.com/huggingface/datasets/tree/master/metrics/ter)
- [BertScore](https://github.com/huggingface/datasets/tree/master/metrics/bertscore)

Run the following command to evaluate the translations:

```bash
accelerate launch eval.py \
--pred_path sample_text/es.txt \
--gold_path sample_text/en2es.translation.m2m100_1.2B.txt 
```

If you want to save the results to a file use the `--output_path` flag.

See [sample_text/en2es.m2m100_1.2B.json](sample_text/en2es.m2m100_1.2B.json) for a sample output.

