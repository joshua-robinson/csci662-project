from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse
import math
import torch


@torch.no_grad()
def get_pythia_ppl(preds: list) -> float:
    ckpt = "EleutherAI/pythia-1b"
    model = AutoModelForCausalLM.from_pretrained(ckpt)
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    if torch.backends.mps.is_available():
        print("Using mps...")
        device = "mps"
    else:
        device = "cpu"
    model = model.to(device)
    model.eval()
    ppls = []
    for pred in tqdm(preds):
        toks = torch.tensor([tokenizer.encode(pred)], device=device)
        loss = model(toks, labels=toks).loss
        ppl = math.exp(loss)
        ppls.append(ppl)
    return sum(ppls) / len(ppls)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--generations_fname",
        help="Path to generations file",
        required=True
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    import json
    args = get_args()
    with open(args.generations_fname, "r") as f:
        contents = f.readlines()
    contents = [json.loads(c) for c in contents]
    preds = []
    for c in contents:
        for completion in c["completions"]:
            preds.append(completion.strip())
    print(get_pythia_ppl(preds=preds))
