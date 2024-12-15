from evaluate import load

import argparse


def get_rougel(preds: list, tgts: list) -> float:
    rouge = load("rouge")
    results = rouge.compute(
        predictions=preds,
        references=tgts
    )
    return results["rougeL"]


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
    tgts = []
    for c in contents:
        for completion in c["completions"]:
            preds.append(completion.strip())
            tgts.append(c["tgt"].strip())
    print(get_rougel(preds=preds, tgts=tgts))
