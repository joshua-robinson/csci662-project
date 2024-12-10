from accelerate.logging import MultiProcessAdapter
from datasets import DatasetDict
from transformers import PreTrainedTokenizer


def strategic_tokenize_function(
    examples: DatasetDict,
    compress: bool,
    tokenizer: PreTrainedTokenizer,
    text_column_name: str,
    max_span_length: int,
    bound_ratio: float,
    cl_token: str,
    cr_token: str,
    logger: MultiProcessAdapter
) -> DatasetDict:
    # if not compress:
    #     return tokenizer(examples[text_column_name])
    # MAX_BOUNDED_LEN = max_span_length
    # avg_length = (2+MAX_BOUNDED_LEN) / 2
    # MAX_BOUNDED_RATIO = bound_ratio # ratio before taking added <CR> into account
    # logger.info(f"compression ratio before: {MAX_BOUNDED_RATIO}")
    # MAX_BOUNDED_RATIO = MAX_BOUNDED_RATIO * (avg_length / (avg_length-1)) # ratio after taking added <CR> into account
    # logger.info(f"compression ratio after: {MAX_BOUNDED_RATIO}")
    # nums = len(examples[text_column_name])
    # for m in range(nums):
    #     cur_text = examples[text_column_name][m]
    #     splitted_text = cur_text.split(" ")
    #     cur_length = len(splitted_text)
    #     if cur_length < 30:
    #         continue
    #     max_bounded_tokens = int(MAX_BOUNDED_RATIO*cur_length)
    #     bounded_tokens = 0
    #     bounded_tokens_indices = set()
    #     boundary = []
    #     avg_times = max_bounded_tokens // (avg_length)
    #     times = 0
    #     while bounded_tokens<max_bounded_tokens:
    #         l = random.randint(0, cur_length-MAX_BOUNDED_LEN-1)
    #         r = l + random.randint(1, MAX_BOUNDED_LEN-1)
    #         f = True
    #         for i in range(l, r+1):
    #             if i in bounded_tokens_indices:
    #                 f = False
    #                 break
    #         if f:
    #             for i in range(l, r+1):
    #                 bounded_tokens_indices.add(i)
    #                 boundary.append((l, r))
    #                 bounded_tokens += r-l+1
    #         times += 1
    #         if times >= 3 * avg_times:
    #             break
    #     boundary = list(sorted(boundary))
    #     final_tokens = []
    #     prev_index = -1
    #     for i, j in boundary:
    #         final_tokens.extend(splitted_text[prev_index+1:i])
    #         final_tokens.extend([cl_token]+splitted_text[i:j+1]+[cr_token])
    #         prev_index = j
    #     final_tokens.extend(splitted_text[prev_index+1:])
    #     final_text = " ".join(final_tokens)
    #     examples[text_column_name][m] = final_text
    # return tokenizer(examples[text_column_name])
    pass
