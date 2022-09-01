from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    DataCollatorForSeq2Seq,
)
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import DatasetReader, count_lines
import os
from accelerate import Accelerator, DistributedType
from accelerate.memory_utils import find_executable_batch_size


def get_dataloader(
    accelerator: Accelerator,
    filename: str,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_length: int,
) -> DataLoader:

    dataset = DatasetReader(filename, tokenizer, max_length)
    if accelerator.distributed_type == DistributedType.TPU:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            padding="max_length",
            max_length=max_length,
            label_pad_token_id=tokenizer.pad_token_id,
            return_tensors="pt",
        )
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            padding=True,
            label_pad_token_id=tokenizer.pad_token_id,
            # max_length=max_length, No need to set max_length here, we already truncate in the preprocess function
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
    )


def main(
    sentences_path: str,
    output_path: str,
    source_lang: str,
    target_lang: str,
    starting_batch_size: int,
    model_name: str = "facebook/m2m100_1.2B",
    cache_dir: str = None,
    precision: str = "32",
    max_length: int = 128,
    num_beams: int = 4,
    num_return_sequences: int = 1,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
):

    if not os.path.exists(os.path.abspath(os.path.dirname(output_path))):
        os.makedirs(os.path.abspath(os.path.dirname(output_path)))

    accelerator = Accelerator(
        mixed_precision=precision if precision != "32" else "no",
        split_batches=False,
        dispatch_batches=False,
    )

    print(f"Loading tokenizer {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name, cache_dir=cache_dir
    )
    print(f"Loading model {model_name}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path=model_name, cache_dir=cache_dir
    )

    model.eval()

    print(f"Preparing data...\n")

    if precision == "32":
        model = model.float()
    elif precision == "fp16":
        model = model.half()
    elif precision == "bf16":
        model = model.bfloat16()
    else:
        raise ValueError("Precision not supported. Supported values: 32, fp16, bf16")

    tokenizer.src_lang = source_lang
    try:
        lang_code_to_idx = tokenizer.lang_code_to_id[target_lang]
    except KeyError:
        raise KeyError(
            f"Language {target_lang} not found in tokenizer. Available languages: {tokenizer.lang_code_to_id.keys()}"
        )

    gen_kwargs = {
        "max_length": max_length,
        "num_beams": num_beams,
        "num_return_sequences": num_return_sequences,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
    }

    # total_lines: int = count_lines(sentences_path)

    if accelerator.is_main_process:
        print(
            f"** Translation **\n"
            f"Input file: {sentences_path}\n"
            f"Output file: {output_path}\n"
            f"Source language: {source_lang}\n"
            f"Target language: {target_lang}\n"
            f"Starting batch size: {starting_batch_size}\n"
            f"Device: {str(accelerator.device).split(':')[0]}\n"
            f"Num. Devices: {accelerator.num_processes}\n"
            f"Distributed_type: {accelerator.distributed_type}\n"
            f"Max length: {max_length}\n"
            f"Precision: {model.dtype}\n"
            f"Model: {model_name}\n"
        )
        print("** Generation parameters **")
        print("\n".join(f"{k}: {v}" for k, v in gen_kwargs.items()))
        print("\n")

    @find_executable_batch_size(starting_batch_size=starting_batch_size)
    def inference(batch_size):
        nonlocal model, tokenizer, sentences_path, max_length, output_path, lang_code_to_idx, gen_kwargs, precision

        print(f"Translating with batch size {batch_size}")

        data_loader = get_dataloader(
            accelerator=accelerator,
            filename=sentences_path,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
        )

        model, data_loader = accelerator.prepare(model, data_loader)

        samples_seen: int = 0

        with tqdm(
            total=len(data_loader.dataset),
            desc="Dataset translation",
            leave=True,
            ascii=True,
            disable=(not accelerator.is_main_process),
        ) as pbar, open(output_path, "w", encoding="utf-8") as output_file:
            with torch.no_grad():
                for step, batch in enumerate(data_loader):
                    batch["input_ids"] = batch["input_ids"]
                    batch["attention_mask"] = batch["attention_mask"]

                    generated_tokens = accelerator.unwrap_model(model).generate(
                        **batch, forced_bos_token_id=lang_code_to_idx, **gen_kwargs
                    )

                    generated_tokens = accelerator.pad_across_processes(
                        generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                    )

                    generated_tokens = (
                        accelerator.gather(generated_tokens).cpu().numpy()
                    )

                    tgt_text = tokenizer.batch_decode(
                        generated_tokens, skip_special_tokens=True
                    )
                    if accelerator.is_main_process:
                        if step == len(data_loader) - 1:
                            tgt_text = tgt_text[
                                : (len(data_loader.dataset) * num_return_sequences)
                                - samples_seen
                            ]
                        else:
                            samples_seen += len(tgt_text)

                        print("\n".join(tgt_text), file=output_file)

                    pbar.update(len(tgt_text) // gen_kwargs["num_return_sequences"])

    inference()
    print(f"Translation done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the translation experiments")
    parser.add_argument(
        "--sentences_path",
        type=str,
        required=True,
        help="Path to a txt file containing the sentences to translate. One sentence per line.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to a txt file where the translated sentences will be written.",
    )

    parser.add_argument(
        "--source_lang",
        type=str,
        required=True,
        help="Source language id. See: supported_languages.md",
    )

    parser.add_argument(
        "--target_lang",
        type=str,
        required=True,
        help="Target language id. See: supported_languages.md",
    )

    parser.add_argument(
        "--starting_batch_size",
        type=int,
        default=128,
        help="Starting batch size, we will automatically reduce it if we find an OOM error."
        "If you use multiple devices, we will divide this number by the number of devices.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/m2m100_1.2B",
        help="Path to the model to use. See: https://huggingface.co/models",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory from which to load the model, or None to not cache",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum number of tokens in the source sentence and generated sentence. "
        "Increase this value to translate longer sentences, at the cost of increasing memory usage.",
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="Number of beams for beam search, m2m10 author recommends 5, but it might use too much memory",
    )

    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of possible translation to return for each sentence (num_return_sequences<=num_beams).",
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        choices=["bf16", "fp16", "32"],
        help="Precision of the model. bf16, fp16 or 32.",
    )

    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling instead of beam search.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature for sampling, value used only if do_sample is True.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="If do_sample is True, will sample from the top k most likely tokens.",
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.75,
        help="If do_sample is True, will sample from the top k most likely tokens.",
    )

    args = parser.parse_args()

    main(
        sentences_path=args.sentences_path,
        output_path=args.output_path,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        starting_batch_size=args.starting_batch_size,
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        max_length=args.max_length,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        precision=args.precision,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
