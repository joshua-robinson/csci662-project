from datasets import load_dataset
from models import CustomGPTNeoXForCausalLM
from peft import PeftModel, PeftModelForCausalLM
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast
)
from typing import Dict, List
from utils import ckpt_associated_with_gptneox

import argparse
import json
import os
import random
import torch
import transformers


class TooManyTokensGeneratedException(Exception):
    pass


def remove_cl_and_cr_from_text(text: str) -> str:
    for symbol in ["<CL>", "<CR>"]:
        text = text.replace(f" {symbol} ", " ")
        text = text.replace(f" {symbol}", " ")
        text = text.replace(f"{symbol} ", " ")
    return text


def tokenize_for_kv_compression(
    text: str,
    max_span_len: int,
    compression_ratio: float,
    tokenizer: PreTrainedTokenizer
) -> List[int]:
        CL_TOKEN = "<CL>"
        CR_TOKEN = "<CR>"

        MAX_BOUNDED_LEN = max_span_len
        avg_length = (2 + MAX_BOUNDED_LEN) / 2

        # ratio before taking added <CR> into account
        MAX_BOUNDED_RATIO = compression_ratio

        # ratio after taking added <CR> into account
        MAX_BOUNDED_RATIO = MAX_BOUNDED_RATIO * (avg_length / (avg_length-1))

        splitted_text = text.split(" ")
        cur_length = len(splitted_text)
        if cur_length < 30:
            return tokenizer(text)
        max_bounded_tokens = int(MAX_BOUNDED_RATIO*cur_length)
        bounded_tokens = 0
        bounded_tokens_indices = set()
        boundary = []
        avg_times = max_bounded_tokens // (avg_length)
        times = 0
        while bounded_tokens < max_bounded_tokens:
            l = random.randint(0, cur_length-MAX_BOUNDED_LEN-1)
            r = l + random.randint(1, MAX_BOUNDED_LEN-1)
            f = True
            for i in range(l, r+1):
                if i in bounded_tokens_indices:
                    f = False
                    break
            if f:
                for i in range(l, r+1):
                    bounded_tokens_indices.add(i)
                boundary.append((l, r))
                bounded_tokens += r-l+1
            times += 1
            if times >= 3 * avg_times:
                break
        boundary = list(sorted(boundary))
        final_tokens = []
        prev_index = -1
        for i, j in boundary:
            final_tokens.extend(splitted_text[prev_index+1:i])
            final_tokens.extend([CL_TOKEN]+splitted_text[i:j+1]+[CR_TOKEN])
            prev_index = j
        final_tokens.extend(splitted_text[prev_index+1:])
        final_text = " ".join(final_tokens)
        return tokenizer(final_text)["input_ids"]


def group_texts_for_kv_compression(
    input_ids: str,
    cl_token_id: int,
    cr_token_id: int
) -> Dict:
    result = {'input_ids': [input_ids]}
    new_attn_mask = [[[0]*len(result['input_ids'][0]) for _ in range(len(result['input_ids'][0]))] for _ in range(len(result['input_ids']))]
    masked_indices_batch = []
    boundary_batch = []
    for i in range(len(result['input_ids'])):
        st = []
        boundary = []
        for j in range(len(result['input_ids'][i])):
            if result['input_ids'][i][j] == cl_token_id:
                st.append(j)
            elif result['input_ids'][i][j] == cr_token_id:
                if len(st)>0:
                    ll = st.pop()
                else: ll = -1
                boundary.append((ll+1, j-1))
        masked_indices = set()
        for l, r in boundary:
            for i in range(l, r+1): masked_indices.add(i)
        masked_indices_batch.append(masked_indices)
        boundary_batch.append(boundary)
    # modify attention mask
    # <CR> token can see its corresponding <CL> token
    # but other tokens cannot see <CL> tokens
    for i in range(len(result['input_ids'])):
        for j in range(len(result['input_ids'][i])):
            for k in range(j, -1, -1):
                if k==j: new_attn_mask[i][j][k] = 1
                elif result['input_ids'][i][j] not in [cl_token_id, cr_token_id]:
                    if result['input_ids'][i][k] in [cr_token_id] or (k not in masked_indices_batch[i] and result['input_ids'][i][k]!=cl_token_id):
                        new_attn_mask[i][j][k] = 1
                    elif result['input_ids'][i][j] != cl_token_id and j in masked_indices_batch[i]:
                        for l, r in boundary_batch[i]:
                            if l<=k<=j<=r:
                                new_attn_mask[i][j][k] = 1
                elif result['input_ids'][i][j] == cr_token_id :
                    for l, r in boundary_batch[i]:
                        if l-1<=k<=r and (r+1)==j:
                            new_attn_mask[i][j][k] = 1
    result['attention_mask'] = new_attn_mask

    # modify position ids
    final_position_ids = []
    for i in range(len(result['input_ids'])):
        position_ids = list(range(len(result['input_ids'][i])))
        final_position_ids.append(position_ids)
        cur_id = 0
        for j in range(len(result['input_ids'][i])):
            if result['input_ids'][i][j] in [cl_token_id, cr_token_id]:
                final_position_ids[i][j] = 0 if j==0 else final_position_ids[i][j-1]
            else:
                final_position_ids[i][j] = cur_id
                cur_id += 1
    result['position_ids'] = final_position_ids

    return result


def is_word_char(c: str, prev_c: str, next_c: str) -> bool:
    if c.isalnum():
        return True
    elif (prev_c is not None) and (next_c is not None):
        if c in ["'", "’"]:
            return (prev_c.isalpha() and next_c.isalpha())
        elif c == ".":
            return (prev_c.isnumeric() and next_c.isnumeric())
    elif next_c is not None:
        if c == "$":
            return next_c.isnumeric()
    else:
        return False


def get_word_indices(text: str) -> List[int]:
    """Returns list of indices denoting where each word in a text starts."""
    word_indices = []
    word_start = 0
    word_len = 0
    for i in range(len(text)):
        if is_word_char(
            c=text[i],
            prev_c=None if i == 0 else text[i-1],
            next_c=None if i == len(text) - 1 else text[i+1]
        ):
            if word_len == 0:
                word_start = i
            word_len += 1
        else:
            if word_len > 0:
                word_indices.append(word_start)
                word_len = 0
    if word_len:
        word_indices.append(word_start)
    return word_indices


def test_tokenizer(tokenizer: PreTrainedTokenizer) -> None:
    for sym in ["<CL>", "<CR>"]:
        if len(tokenizer.encode(sym)) != 1:
            raise ValueError(f"Tokenizer vocab does not include {sym}")


def get_word_count(text: str) -> int:
    return len(get_word_indices(text=text))


def get_through_kth_word(text: str, k: int) -> str:
    """Return the part of the string up to the end of the k-th word."""
    if k <= 0:
        raise ValueError("k must be positive")
    word_indices = get_word_indices(text=text)
    try:
        i = word_indices[k-1]
    except IndexError:
        raise ValueError("Not enough words in text")
    while i < len(text) and is_word_char(
        c=text[i],
        prev_c=None if i == 0 else text[i-1],
        next_c=None if i == len(text) - 1 else text[i+1]
    ):
        i += 1
    return text[:i]


def prep_docs(docs: List[str], pfx_len: int, tgt_len: int) -> List[str]:
    """Break each document into a part containing the first pfx_len words 
    and a part containing the next trt_len words."""
    processed_docs = []
    for doc in docs:
        pfx = get_through_kth_word(text=doc, k=pfx_len)
        tgt = get_through_kth_word(text=doc[len(pfx):], k=tgt_len)
        processed_docs.append({"pfx": pfx, "tgt": tgt})
    return processed_docs


def load_docs(fname: str) -> List[str]:
    if not fname.endswith("json"):
        raise ValueError("Docs file must be JSON file")
    ds = load_dataset("json", data_files=fname, split="train")
    docs = []
    for sample in ds:
        if len(sample.keys()) > 1:
            raise ValueError("Each document should be associated with one key")
        docs.append(sample[list(sample.keys())[0]])
    return docs


def load_model(model_dir: str) -> PeftModelForCausalLM:
    with open(f"{model_dir}/config.json", "r") as f:
        config = json.load(f)
    if ckpt_associated_with_gptneox(ckpt=config["_name_or_path"]):
        model = CustomGPTNeoXForCausalLM.from_pretrained(model_dir)
        model = PeftModel.from_pretrained(model, model_dir)
        return model
    else:
        raise NotImplementedError


def do_nucleus_sample(probs: torch.tensor, p: float) -> int:
    sorted_probs, sorted_idxs = torch.sort(probs, descending=True)
    # What is the rightmost index where p could be placed while maintaining
    # sorted order?
    cutoff_idx = torch.searchsorted(
        torch.cumsum(sorted_probs, dim=-1), p, right=True
    )
    if cutoff_idx == 0:
        sampled_idx = 0
    else:
        sorted_probs[cutoff_idx:] = 0
        sampled_idx = torch.multinomial(sorted_probs, 1).item()
    return sorted_idxs[sampled_idx]


@torch.no_grad()
def get_kv_compression_generation(
    prefix: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    n_words_to_generate: int,
    compression_ratio: float,
    nucleus_sampling_p: float,
    ids_to_not_generate: List[int],
    max_tokens: int,
    max_span_len: int
) -> str:
    n_words_at_start = get_word_count(prefix)
    tot_words_tgt = n_words_at_start + n_words_to_generate
    clean_ids = tokenizer.encode(prefix)

    while get_word_count(tokenizer.decode(clean_ids)) <= tot_words_tgt:
        if len(clean_ids) > max_tokens:
            raise TooManyTokensGeneratedException

        all_ids = tokenize_for_kv_compression(
            text=tokenizer.decode(clean_ids),
            max_span_len=max_span_len,
            compression_ratio=compression_ratio,
            tokenizer=tokenizer
        )

        # Handle the prefix using attention mask
        grouped = group_texts_for_kv_compression(
            input_ids=all_ids,
            cl_token_id=tokenizer.encode("<CL>"),
            cr_token_id=tokenizer.encode("<CR>")
        )

        out = model(
            input_ids=torch.tensor([all_ids], device=model.device),
            attention_mask=torch.tensor(grouped["attention_mask"], device=model.device),
            position_ids=torch.tensor(grouped["position_ids"], device=model.device),
        )

        next_token_logits = out.logits[0, -1]
        logit_mask_val = torch.finfo(next_token_logits.dtype).min
        if len(ids_to_not_generate):
            next_token_logits[ids_to_not_generate] = logit_mask_val
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        pred_id = do_nucleus_sample(
            probs=next_token_probs,
            p=nucleus_sampling_p
        ).item()
        clean_ids.append(pred_id)

    text = tokenizer.decode(clean_ids)
    text = get_through_kth_word(text, tot_words_tgt)
    text = text[len(prefix):]
    return text


@torch.no_grad()
def get_local_attention_generation(
    prefix: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    n_words_to_generate: int,
    compression_ratio: float,
    nucleus_sampling_p: float,
    ids_to_not_generate: List[int],
    max_tokens: int
) -> str:
    n_words_at_start = get_word_count(prefix)
    all_ids = tokenizer.encode(prefix)
    tot_words_tgt = n_words_at_start + n_words_to_generate

    # Handle the prefix using attention mask
    ids = torch.tensor([all_ids], device=model.device)
    seq_len = ids.shape[1]
    attn_mask = torch.zeros((seq_len, seq_len), device=model.device)
    for i in range(seq_len):
        for j in range(i+1):
            if j >= int(i*compression_ratio):
                attn_mask[i][j] = 1
    attn_mask = attn_mask.unsqueeze(0)
    out = model(
        input_ids=ids,
        attention_mask=attn_mask,
        use_cache=True
    )

    next_token_logits = out.logits[0, -1]
    logit_mask_val = torch.finfo(next_token_logits.dtype).min
    if len(ids_to_not_generate):
        next_token_logits[ids_to_not_generate] = logit_mask_val
    next_token_probs = torch.softmax(next_token_logits, dim=-1)
    pred_id = do_nucleus_sample(
        probs=next_token_probs,
        p=nucleus_sampling_p
    ).item()
    all_ids.append(pred_id)
    past_key_values = out.past_key_values
    
    # Handle the generation using KV cache
    # On the next line it is <= because we are going to over-generate
    # and then truncate to ensure we don't end with a partial word
    while get_word_count(tokenizer.decode(all_ids)) <= tot_words_tgt:
        if len(all_ids) > max_tokens:
            raise TooManyTokensGeneratedException
        # Compress the cache
        seq_len = len(all_ids)
        attn_mask = torch.zeros((1, seq_len), device=model.device)
        i = seq_len - 1
        for j in range(len(all_ids)):
            if j >= int(i*compression_ratio):
                attn_mask[0][j] = 1
        ids = torch.tensor([all_ids[-1:]], device=model.device)
        new_past_key_values = past_key_values
        out = model(
            input_ids=ids,
            past_key_values=new_past_key_values,
            use_cache=True,
            attention_mask=attn_mask
        )

        next_token_logits = out.logits[0, -1]
        if len(ids_to_not_generate):
            next_token_logits[ids_to_not_generate] = logit_mask_val
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        pred_id = do_nucleus_sample(
            probs=next_token_probs,
            p=nucleus_sampling_p
        ).item()
        all_ids.append(pred_id)
        past_key_values = out.past_key_values
    
    text = get_through_kth_word(tokenizer.decode(all_ids), tot_words_tgt)
    return text[len(prefix):]


def load_tokenizer(model_dir: str) -> PreTrainedTokenizerFast:
    tokenizer_file = f"{model_dir}/tokenizer.json"
    return PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)


def get_args() -> argparse.Namespace:
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_dir",
        help="Model directory resulting from training",
        required=True
    )
    parser.add_argument(
        "-t",
        "--compression_type",
        help="Type of KV cache compression to use",
        choices=["kv", "local"],
        required=True
    )
    parser.add_argument(
        "-r",
        "--compression_ratio",
        help="Compression ratio to use for generation",
        type=float,
        required=True
    )
    parser.add_argument(
        "-o",
        "--out_fname",
        help="File to save generations to (must be .json)",
        required=True
    )
    parser.add_argument(
        "-f",
        "--docs_fname",
        help="Name of file containing documents to use for prefixes, targets",
        default=f"{cur_dir}/c4_sample.json"
    )
    parser.add_argument(
        "-x",
        "--pfx_len",
        help="Number of words generation prefix should have",
        type=int,
        default=128
    )
    parser.add_argument(
        "-g",
        "--gen_len",
        help="Number of words to generate",
        type=int,
        default=64
    )
    parser.add_argument(
        "-p",
        "--nucleus_sampling_p",
        help="Value of p to use for nucleus sampling",
        type=float,
        default=0.9
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="Seed for reproducibility",
        type=int,
        default=0
    )
    parser.add_argument(
        "-b",
        "--blocked_ids",
        nargs="*",
        type=int,
        help="IDs of tokens that should never be generated"
    )
    parser.add_argument(
        "-n",
        "--gens_per_prefix",
        help="Number of completions to generate per prefix",
        type=int,
        default=8
    )
    parser.add_argument(
        "-z",
        "--max_tokens",
        help="If this token count is hit, generation will raise exception",
        type=int,
        default=480
    )
    parser.add_argument(
        "-l",
        "--max_span_len",
        help="Max span len when randomly assigning <CL> and <CR> spans",
        type=int,
        default=25
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    import time

    setup_start_time = time.time()

    # Get command line args
    args = get_args()
    if not args.out_fname.endswith(".json"):
        raise ValueError("out_fname argument must end with .json")

    # Set seed for reproducibility
    random.seed(args.seed)
    transformers.set_seed(args.seed)
    torch.manual_seed(args.seed)

    # Load documents
    docs = load_docs(fname=args.docs_fname)
    prepped = prep_docs(
        docs=docs,
        pfx_len=args.pfx_len,
        tgt_len=args.gen_len
    )

    # Load tokenizer
    tokenizer = load_tokenizer(model_dir=args.model_dir)
    test_tokenizer(tokenizer=tokenizer)

    setup_time = time.time() - setup_start_time
    print("Setup took", setup_time, "seconds")

    # Load model
    loading_model_start_time = time.time()
    model = load_model(model_dir=args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    loading_model_time = time.time() - loading_model_start_time
    print("Loading model took", loading_model_time, "seconds")

    generation_start_time = time.time()

    if args.compression_type == "local":
        gen_fn = get_local_attention_generation
    else:
        gen_fn = get_kv_compression_generation

    completions = []
    for pfx_and_tgt in tqdm(prepped):
        pfx_completions = []
        while len(pfx_completions) < args.gens_per_prefix:
            try:
                fn_args = {
                    "prefix": pfx_and_tgt["pfx"],
                    "model": model,
                    "tokenizer": tokenizer,
                    "n_words_to_generate": args.gen_len,
                    "compression_ratio": args.compression_ratio,
                    "nucleus_sampling_p": args.nucleus_sampling_p,
                    "ids_to_not_generate": args.blocked_ids or [],
                    "max_tokens": args.max_tokens
                }
                if args.compression_type == "kv":
                    fn_args["max_span_len"] = args.max_span_len
                completion = gen_fn(**fn_args)
                pfx_completions.append(completion)
            except TooManyTokensGeneratedException:
                print("Retrying generation...")
                pass
        completions.append(pfx_completions)

    generation_time = time.time() - generation_start_time
    print("Generation took", generation_time, "seconds")

    with open(args.out_fname, "w") as f:
        for i in range(len(prepped)):
            data = {
                "pfx": prepped[i]["pfx"],
                "tgt": prepped[i]["tgt"],
                "completions": completions[i]
            }
            f.write(f"{json.dumps(data)}\n")
