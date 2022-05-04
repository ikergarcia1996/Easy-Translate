
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

Easy-translate is a script for translating large text files in your machine using the [M2M100 models](https://arxiv.org/pdf/2010.11125.pdf) from Facebook/Meta AI.  We also privide a [script](#evaluate-translations) for Easy-Evaluation of your translations 🥳

**M2M100** is a multilingual encoder-decoder (seq-to-seq) model trained for Many-to-Many multilingual translation introduced in this [paper](https://arxiv.org/abs/2010.11125) and first released in [this](https://github.com/pytorch/fairseq/tree/master/examples/m2m_100) repository.

>M2M100 can directly translate between 9,900 directions of 100 languages.

Easy-Translate is built on top of 🤗HuggingFace's [Transformers](https://huggingface.co/docs/transformers/index) and 🤗HuggingFace's [Accelerate](https://huggingface.co/docs/accelerate/index) library.

We currently support:

- CPU / GPU / multi-GPU / TPU acceleration
- BF16 / FP16 / FP32 precision.
- Automatic batch size finder: Forget CUDA OOM errors. Set an initial batch size, if it doesn't fit, we will automatically adjust it.
- Sharded Data Parallel to load huge models sharded on multiple GPUs (See: <https://huggingface.co/docs/accelerate/fsdp>).

>Test the 🔌 Online Demo here: <https://huggingface.co/spaces/Iker/Translate-100-languages>


## Supported languages

See the [Supported languages table](supported_languages.md) for a table of the supported languages and their ids.

**List of supported languages:**
Afrikaans, Amharic, Arabic, Asturian, Azerbaijani, Bashkir, Belarusian, Bulgarian, Bengali, Breton, Bosnian, Catalan, Cebuano, Czech, Welsh, Danish, German, Greeek, English, Spanish, Estonian, Persian, Fulah, Finnish, French, WesternFrisian, Irish, Gaelic, Galician, Gujarati, Hausa, Hebrew, Hindi, Croatian, Haitian, Hungarian, Armenian, Indonesian, Igbo, Iloko, Icelandic, Italian, Japanese, Javanese, Georgian, Kazakh, CentralKhmer, Kannada, Korean, Luxembourgish, Ganda, Lingala, Lao, Lithuanian, Latvian, Malagasy, Macedonian, Malayalam, Mongolian, Marathi, Malay, Burmese, Nepali, Dutch, Norwegian, NorthernSotho, Occitan, Oriya, Panjabi, Polish, Pushto, Portuguese, Romanian, Russian, Sindhi, Sinhala, Slovak, Slovenian, Somali, Albanian, Serbian, Swati, Sundanese, Swedish, Swahili, Tamil, Thai, Tagalog, Tswana, Turkish, Ukrainian, Urdu, Uzbek, Vietnamese, Wolof, Xhosa, Yiddish, Yoruba, Chinese, Zulu

## Supported Models

- **Facebook/m2m100_418M**: <https://huggingface.co/facebook/m2m100_418M>

- **Facebook/m2m100_1.2B**: <https://huggingface.co/facebook/m2m100_1.2B>

- **Facebook/m2m100_12B**: <https://huggingface.co/facebook/m2m100-12B-avg-5-ckpt>

- Any other m2m100 model from HuggingFace's Hub: <https://huggingface.co/models?search=m2m100>

## Requirements

```
Pytorch >= 1.10.0
See: https://pytorch.org/get-started/locally/

Accelerate >= 0.7.1
pip install --upgrade accelerate

HuggingFace Transformers 
pip install --upgrade transformers
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

