import numpy as np
import torch
import torch.nn.functional as F
from accelerate.logging import MultiProcessAdapter
from datasets import DatasetDict, Dataset, load_dataset
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM
import logging

# Load the Wikitext-2 dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Ensure validation split is present
validation_split_percentage = 10  # Define the percentage for the validation split
if "validation" not in dataset.keys():
    print("Validation split not found. Creating a validation split...")
    dataset["validation"] = dataset["train"].train_test_split(
        test_size=validation_split_percentage / 100, shuffle=True, seed=42
    )["test"]
    dataset["train"] = dataset["train"].train_test_split(
        test_size=validation_split_percentage / 100, shuffle=True, seed=42
    )["train"]

print(f"Dataset splits: {list(dataset.keys())}")
print(f"Train size: {len(dataset['train'])}")
print(f"Validation size: {len(dataset['validation'])}")

# Select 5 examples from the training set for demonstration
dataset = dataset["train"].select(range(5))

def strategic_tokenize_function(
    examples,
    compress: bool,
    tokenizer: PreTrainedTokenizer,
    text_column_name: str,
    max_span_length: int,
    bound_ratio: float,
    cl_token: str,
    cr_token: str,
    logger: MultiProcessAdapter,
    base_model_for_si=None,
    lexical_unit="sentence"
) -> dict:
    if not compress:
        encoded = tokenizer(
            examples[text_column_name],
            padding=True,
            truncation=True,
            max_length=128
        )
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}

    processed_texts = []

    def compute_self_information(tokens):
        inputs = tokenizer(tokens, return_tensors='pt', add_special_tokens=False, is_split_into_words=True)
        input_ids = inputs["input_ids"]  # Shape: [1, seq_len]

        if base_model_for_si is None:
            return np.random.uniform(0, 10, size=len(tokens))

        with torch.no_grad():
            outputs = base_model_for_si(**inputs)
        logits = outputs.logits  # [1, seq_len, vocab_size]
        probs = F.softmax(logits, dim=-1)

        si_values = []
        token_ids = input_ids[0]
        for i in range(len(tokens)):
            token_id = token_ids[i].item()
            p = probs[0, i, token_id].item()
            if p <= 0:
                p = 1e-12
            si = -np.log2(p)
            si_values.append(si)

        return si_values

    def segment_lexical_units(tokens, mode="sentence"):
        if mode == "sentence":
            text = " ".join(tokens)
            sentences = text.split('.')
            units = []
            for s in sentences:
                s = s.strip()
                if s:
                    unit_tokens = s.split()
                    units.append(unit_tokens)
            return units
        elif mode == "token":
            return [[t] for t in tokens]
        else:
            return [[t] for t in tokens]

    for text in examples[text_column_name]:
        tokens = text.split()
        if len(tokens) < 30:
            processed_texts.append(text)
            continue

        si_values = compute_self_information(tokens)
        units = segment_lexical_units(tokens, lexical_unit)

        unit_si = []
        start_idx = 0
        for u in units:
            length = len(u)
            si_sum = sum(si_values[start_idx:start_idx+length])
            unit_si.append(si_sum)
            start_idx += length

        threshold = np.percentile(unit_si, 100 * (1 - bound_ratio))
        retained_units = [u for u, si_val in zip(units, unit_si) if si_val >= threshold]

        if len(retained_units) == 0:
            final_text = text
        else:
            final_tokens = []
            for u in retained_units:
                final_tokens.extend(u)
            final_text = " ".join(final_tokens)

        processed_texts.append(final_text)

    encoded = tokenizer(
        processed_texts,
        padding=True,
        truncation=True,
        max_length=128
    )
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}

if __name__ == "__main__":
    base_logger = logging.getLogger(__name__)
    base_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    base_logger.addHandler(handler)
    logger = MultiProcessAdapter(base_logger)

    # Initialize the tokenizer with add_prefix_space=True
        # Initialize the tokenizer with add_prefix_space=True
    tokenizer = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)

    # Set the pad token (use EOS token as pad token)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("gpt2")

    compress = True
    text_column_name = "text"
    max_span_length = 10
    bound_ratio = 0.5
    cl_token = "<CL>"
    cr_token = "<CR>"
    lexical_unit = "sentence"

    tokenized = dataset.map(
        strategic_tokenize_function,
        batched=True,
        fn_kwargs={
            "compress": compress,
            "tokenizer": tokenizer,
            "text_column_name": text_column_name,
            "max_span_length": max_span_length,
            "bound_ratio": bound_ratio,
            "cl_token": cl_token,
            "cr_token": cr_token,
            "logger": logger,
            "base_model_for_si": model,
            "lexical_unit": lexical_unit
        },
        remove_columns=[text_column_name]
    )

    print(tokenized["input_ids"])
