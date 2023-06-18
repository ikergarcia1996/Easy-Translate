from dataset import ParallelTextReader
from torch.utils.data import DataLoader
from accelerate import find_executable_batch_size
from evaluate import load
from tqdm import tqdm
import torch
import json
import argparse
import numpy as np
import os


def get_dataloader(pred_path: str, gold_path: str, batch_size: int):
    """
    Returns a dataloader for the given files.
    """

    def collate_fn(batch):
        return list(map(list, zip(*batch)))

    reader = ParallelTextReader(pred_path=pred_path, gold_path=gold_path)
    dataloader = DataLoader(
        reader, batch_size=batch_size, collate_fn=collate_fn, num_workers=0
    )
    return dataloader


def eval_files(
    pred_path: str,
    gold_path: str,
    bert_score_model: str,
    starting_batch_size: int = 128,
    output_path: str = None,
):
    """
    Evaluates the given files.
    """
    if torch.cuda.is_available():
        device = "cuda:0"
        print("We will use a GPU to calculate BertScore.")
    else:
        device = "cpu"
        print(
            f"We will use the CPU to calculate BertScore, this can be slow for large datasets."
        )

    dataloader = get_dataloader(pred_path, gold_path, starting_batch_size)
    print("Loading sacrebleu...")
    sacrebleu = load("sacrebleu")
    print("Loading rouge...")
    rouge = load("rouge")
    print("Loading bleu...")
    bleu = load("bleu")
    print("Loading meteor...")
    meteor = load("meteor")
    print("Loading ter...")
    ter = load("ter")
    print("Loading BertScore...")
    bert_score = load("bertscore")

    with tqdm(total=len(dataloader.dataset), desc="Loading data...") as pbar:
        for predictions, references in dataloader:
            sacrebleu.add_batch(predictions=predictions, references=references)
            rouge.add_batch(predictions=predictions, references=references)
            bleu.add_batch(predictions=predictions, references=references)
            meteor.add_batch(predictions=predictions, references=references)
            ter.add_batch(predictions=predictions, references=references)
            bert_score.add_batch(predictions=predictions, references=references)
            pbar.update(len(predictions))

    result_dictionary = {"path": pred_path}
    print("Computing sacrebleu")
    result_dictionary["sacrebleu"] = sacrebleu.compute()
    print("Computing rouge score")
    result_dictionary["rouge"] = rouge.compute(
        use_aggregator=True, rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"]
    )
    print("Computing bleu score")
    result_dictionary["bleu"] = bleu.compute()
    print("Computing meteor score")
    result_dictionary["meteor"] = meteor.compute()
    print("Computing ter score")
    result_dictionary["ter"] = ter.compute()

    @find_executable_batch_size(starting_batch_size=starting_batch_size)
    def inference(batch_size):
        nonlocal bert_score, bert_score_model
        print(f"Computing bert score with batch size {batch_size} on {device}")
        results = bert_score.compute(
            model_type=bert_score_model,
            batch_size=batch_size,
            device=device,
            use_fast_tokenizer=True,
        )

        results["precision"] = np.average(results["precision"])
        results["recall"] = np.average(results["recall"])
        results["f1"] = np.average(results["f1"])

        return results

    result_dictionary["bert_score"] = inference()

    if output_path is not None:
        if not os.path.exists(os.path.abspath(os.path.dirname(output_path))):
            os.makedirs(os.path.abspath(os.path.dirname(output_path)))
        with open(output_path, "w") as f:
            json.dump(result_dictionary, f, indent=4)

    print(f"Results: {json.dumps(result_dictionary,indent=4)}")

    return result_dictionary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the translation evaluation experiments"
    )
    parser.add_argument(
        "--pred_path",
        type=str,
        required=True,
        help="Path to a txt file containing the predicted sentences.",
    )

    parser.add_argument(
        "--gold_path",
        type=str,
        required=True,
        help="Path to a txt file containing the gold sentences.",
    )

    parser.add_argument(
        "--starting_batch_size",
        type=int,
        default=64,
        help="Starting batch size for BertScore, we will automatically reduce it if we find an OOM error.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to a json file to save the results. If not given, the results will be printed to the console.",
    )

    parser.add_argument(
        "--bert_score_model",
        type=str,
        default="microsoft/deberta-xlarge-mnli",
        help="Model to use for BertScore. See: https://github.com/huggingface/datasets/tree/master/metrics/bertscore"
        "and https://github.com/Tiiiger/bert_score for more details.",
    )

    args = parser.parse_args()

    eval_files(
        pred_path=args.pred_path,
        gold_path=args.gold_path,
        starting_batch_size=args.starting_batch_size,
        output_path=args.output_path,
        bert_score_model=args.bert_score_model,
    )
