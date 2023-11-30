# Run with 'python -m unittest tests.test_translation'

import unittest
import tempfile
import os
from translate import main
import transformers


class Inputs(unittest.TestCase):
    def test_m2m100_inputs(self):
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Create a temporary file

            input_path = os.path.join(tmpdirname, "source.txt")
            output_path = os.path.join(tmpdirname, "target.txt")

            with open(
                os.path.join(tmpdirname, "source.txt"), "w", encoding="utf8"
            ) as f:
                print("Hello, world, my name is Iker!", file=f)

            main(
                sentences_path=input_path,
                sentences_dir=None,
                files_extension="txt",
                output_path=output_path,
                source_lang="en",
                target_lang="es",
                starting_batch_size=32,
                model_name="facebook/m2m100_418M",
                lora_weights_name_or_path=None,
                force_auto_device_map=True,
                precision=None,
                max_length=64,
                num_beams=2,
                num_return_sequences=1,
                do_sample=False,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
                keep_special_tokens=False,
                keep_tokenization_spaces=False,
                repetition_penalty=None,
                prompt=None,
            )

            main(
                sentences_path=None,
                sentences_dir=tmpdirname,
                files_extension="txt",
                output_path=os.path.join(tmpdirname, "target"),
                source_lang="en",
                target_lang="es",
                starting_batch_size=32,
                model_name="facebook/m2m100_418M",
                lora_weights_name_or_path=None,
                force_auto_device_map=True,
                precision=None,
                max_length=64,
                num_beams=2,
                num_return_sequences=1,
                do_sample=False,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
                keep_special_tokens=False,
                keep_tokenization_spaces=False,
                repetition_penalty=None,
                prompt=None,
            )


class Translations(unittest.TestCase):
    def test_m2m100(self):
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Create a temporary file

            input_path = os.path.join(tmpdirname, "source.txt")
            output_path = os.path.join(tmpdirname, "target.txt")

            with open(
                os.path.join(tmpdirname, "source.txt"), "w", encoding="utf8"
            ) as f:
                print("Hello, world, my name is Iker!", file=f)

            model_name = "facebook/m2m100_418M"
            src_lang = "en"
            tgt_lang = "es"

            main(
                sentences_path=input_path,
                sentences_dir=None,
                files_extension="txt",
                output_path=output_path,
                source_lang=src_lang,
                target_lang=tgt_lang,
                starting_batch_size=32,
                model_name=model_name,
                lora_weights_name_or_path=None,
                force_auto_device_map=True,
                precision="bf16",
                max_length=64,
                num_beams=2,
                num_return_sequences=1,
                do_sample=False,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
                keep_special_tokens=False,
                keep_tokenization_spaces=False,
                repetition_penalty=None,
                prompt=None,
            )

            main(
                sentences_path=input_path,
                sentences_dir=None,
                files_extension="txt",
                output_path=output_path,
                source_lang=src_lang,
                target_lang=tgt_lang,
                starting_batch_size=32,
                model_name=model_name,
                lora_weights_name_or_path=None,
                force_auto_device_map=True,
                precision="4",
                max_length=64,
                num_beams=2,
                num_return_sequences=1,
                do_sample=False,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
                keep_special_tokens=False,
                keep_tokenization_spaces=False,
                repetition_penalty=None,
                prompt=None,
            )

    def test_nllb200(self):
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Create a temporary file

            input_path = os.path.join(tmpdirname, "source.txt")
            output_path = os.path.join(tmpdirname, "target.txt")

            with open(
                os.path.join(tmpdirname, "source.txt"), "w", encoding="utf8"
            ) as f:
                print("Hello, world, my name is Iker!", file=f)

            model_name = "facebook/nllb-200-distilled-600M"
            src_lang = "eng_Latn"
            tgt_lang = "spa_Latn"

            main(
                sentences_path=input_path,
                sentences_dir=None,
                files_extension="txt",
                output_path=output_path,
                source_lang=src_lang,
                target_lang=tgt_lang,
                starting_batch_size=32,
                model_name=model_name,
                lora_weights_name_or_path=None,
                force_auto_device_map=True,
                precision="bf16",
                max_length=64,
                num_beams=2,
                num_return_sequences=1,
                do_sample=False,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
                keep_special_tokens=False,
                keep_tokenization_spaces=False,
                repetition_penalty=None,
                prompt=None,
            )

            main(
                sentences_path=input_path,
                sentences_dir=None,
                files_extension="txt",
                output_path=output_path,
                source_lang=src_lang,
                target_lang=tgt_lang,
                starting_batch_size=32,
                model_name=model_name,
                lora_weights_name_or_path=None,
                force_auto_device_map=True,
                precision="4",
                max_length=64,
                num_beams=2,
                num_return_sequences=1,
                do_sample=False,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
                keep_special_tokens=False,
                keep_tokenization_spaces=False,
                repetition_penalty=None,
                prompt=None,
            )

    def test_mbart(self):
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Create a temporary file

            input_path = os.path.join(tmpdirname, "source.txt")
            output_path = os.path.join(tmpdirname, "target.txt")

            with open(
                os.path.join(tmpdirname, "source.txt"), "w", encoding="utf8"
            ) as f:
                print("Hello, world, my name is Iker!", file=f)

            model_name = "facebook/mbart-large-50"
            src_lang = "en_XX"
            tgt_lang = "es_XX"

            main(
                sentences_path=input_path,
                sentences_dir=None,
                files_extension="txt",
                output_path=output_path,
                source_lang=src_lang,
                target_lang=tgt_lang,
                starting_batch_size=32,
                model_name=model_name,
                lora_weights_name_or_path=None,
                force_auto_device_map=True,
                precision="bf16",
                max_length=64,
                num_beams=2,
                num_return_sequences=1,
                do_sample=False,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
                keep_special_tokens=False,
                keep_tokenization_spaces=False,
                repetition_penalty=None,
                prompt=None,
            )

            main(
                sentences_path=input_path,
                sentences_dir=None,
                files_extension="txt",
                output_path=output_path,
                source_lang=src_lang,
                target_lang=tgt_lang,
                starting_batch_size=32,
                model_name=model_name,
                lora_weights_name_or_path=None,
                force_auto_device_map=True,
                precision="4",
                max_length=64,
                num_beams=2,
                num_return_sequences=1,
                do_sample=False,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
                keep_special_tokens=False,
                keep_tokenization_spaces=False,
                repetition_penalty=None,
                prompt=None,
            )

    def test_opus(self):
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Create a temporary file

            input_path = os.path.join(tmpdirname, "source.txt")
            output_path = os.path.join(tmpdirname, "target.txt")

            with open(
                os.path.join(tmpdirname, "source.txt"), "w", encoding="utf8"
            ) as f:
                print("Hello, world, my name is Iker!", file=f)

            model_name = "Helsinki-NLP/opus-mt-en-es"
            src_lang = None
            tgt_lang = None

            main(
                sentences_path=input_path,
                sentences_dir=None,
                files_extension="txt",
                output_path=output_path,
                source_lang=src_lang,
                target_lang=tgt_lang,
                starting_batch_size=32,
                model_name=model_name,
                lora_weights_name_or_path=None,
                force_auto_device_map=False,
                precision="bf16",
                max_length=64,
                num_beams=2,
                num_return_sequences=1,
                do_sample=False,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
                keep_special_tokens=False,
                keep_tokenization_spaces=False,
                repetition_penalty=None,
                prompt=None,
            )

            main(
                sentences_path=input_path,
                sentences_dir=None,
                files_extension="txt",
                output_path=output_path,
                source_lang=src_lang,
                target_lang=tgt_lang,
                starting_batch_size=32,
                model_name=model_name,
                lora_weights_name_or_path=None,
                force_auto_device_map=False,
                precision="4",
                max_length=64,
                num_beams=2,
                num_return_sequences=1,
                do_sample=False,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
                keep_special_tokens=False,
                keep_tokenization_spaces=False,
                repetition_penalty=None,
                prompt=None,
            )

    @unittest.skipIf(
        transformers.__version__ > "4.34.0",
        "Small100 tokenizer is not supported in transformers > 4.34.0. Please use transformers <= 4.34.0 if you want to use small100",
    )
    def test_small100(self):
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Create a temporary file

            input_path = os.path.join(tmpdirname, "source.txt")
            output_path = os.path.join(tmpdirname, "target.txt")

            with open(
                os.path.join(tmpdirname, "source.txt"), "w", encoding="utf8"
            ) as f:
                print("Hello, world, my name is Iker!", file=f)

            model_name = "alirezamsh/small100"
            src_lang = None
            tgt_lang = "es"

            main(
                sentences_path=input_path,
                sentences_dir=None,
                files_extension="txt",
                output_path=output_path,
                source_lang=src_lang,
                target_lang=tgt_lang,
                starting_batch_size=32,
                model_name=model_name,
                lora_weights_name_or_path=None,
                force_auto_device_map=True,
                precision="bf16",
                max_length=64,
                num_beams=2,
                num_return_sequences=1,
                do_sample=False,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
                keep_special_tokens=False,
                keep_tokenization_spaces=False,
                repetition_penalty=None,
                prompt=None,
            )

            main(
                sentences_path=input_path,
                sentences_dir=None,
                files_extension="txt",
                output_path=output_path,
                source_lang=src_lang,
                target_lang=tgt_lang,
                starting_batch_size=32,
                model_name=model_name,
                lora_weights_name_or_path=None,
                force_auto_device_map=True,
                precision="4",
                max_length=64,
                num_beams=2,
                num_return_sequences=1,
                do_sample=False,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
                keep_special_tokens=False,
                keep_tokenization_spaces=False,
                repetition_penalty=None,
                prompt=None,
            )

    def test_seamless(self):
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Create a temporary file

            input_path = os.path.join(tmpdirname, "source.txt")
            output_path = os.path.join(tmpdirname, "target.txt")

            with open(
                os.path.join(tmpdirname, "source.txt"), "w", encoding="utf8"
            ) as f:
                print("Hello, world, my name is Iker!", file=f)

            model_name = "facebook/hf-seamless-m4t-medium"
            src_lang = "eng"
            tgt_lang = "spa"

            main(
                sentences_path=input_path,
                sentences_dir=None,
                files_extension="txt",
                output_path=output_path,
                source_lang=src_lang,
                target_lang=tgt_lang,
                starting_batch_size=32,
                model_name=model_name,
                lora_weights_name_or_path=None,
                force_auto_device_map=True,
                precision="bf16",
                max_length=64,
                num_beams=2,
                num_return_sequences=1,
                do_sample=False,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
                keep_special_tokens=False,
                keep_tokenization_spaces=False,
                repetition_penalty=None,
                prompt=None,
            )

            main(
                sentences_path=input_path,
                sentences_dir=None,
                files_extension="txt",
                output_path=output_path,
                source_lang=src_lang,
                target_lang=tgt_lang,
                starting_batch_size=32,
                model_name=model_name,
                lora_weights_name_or_path=None,
                force_auto_device_map=True,
                precision="4",
                max_length=64,
                num_beams=2,
                num_return_sequences=1,
                do_sample=False,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
                keep_special_tokens=False,
                keep_tokenization_spaces=False,
                repetition_penalty=None,
                prompt=None,
            )


class Prompting(unittest.TestCase):
    def test_llama(self):
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Create a temporary file

            input_path = os.path.join(tmpdirname, "source.txt")
            output_path = os.path.join(tmpdirname, "target.txt")

            with open(
                os.path.join(tmpdirname, "source.txt"), "w", encoding="utf8"
            ) as f:
                print("Hello, world, my name is Iker!", file=f)

            model_name = "stas/tiny-random-llama-2"
            prompt = "Translate English to Spanish: %%SENTENCE%%"

            main(
                sentences_path=input_path,
                sentences_dir=None,
                files_extension="txt",
                output_path=output_path,
                source_lang=None,
                target_lang=None,
                starting_batch_size=32,
                model_name=model_name,
                lora_weights_name_or_path=None,
                force_auto_device_map=True,
                precision="bf16",
                max_length=64,
                num_beams=2,
                num_return_sequences=1,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
                keep_special_tokens=False,
                keep_tokenization_spaces=False,
                repetition_penalty=None,
                prompt=prompt,
            )

            main(
                sentences_path=input_path,
                sentences_dir=None,
                files_extension="txt",
                output_path=output_path,
                source_lang=None,
                target_lang=None,
                starting_batch_size=32,
                model_name=model_name,
                lora_weights_name_or_path=None,
                force_auto_device_map=True,
                precision="4",
                max_length=64,
                num_beams=2,
                num_return_sequences=1,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
                keep_special_tokens=False,
                keep_tokenization_spaces=False,
                repetition_penalty=None,
                prompt=prompt,
            )
