from accelerate.logging import MultiProcessAdapter
from datasets import DatasetDict
from transformers import PreTrainedTokenizer
from collections import deque
from datasets import Dataset
# from selective context code
import selective_context
from selective_context import SelectiveContext


def find_word_level_indexes(words, phrases):
    idx = 0
    phrase_locs = []

    for phrase in phrases:  # Use a set to handle duplicate phrases
        # Tokenize the phrase into words
        phrase_words = phrase.split()
        phrase_len = len(phrase_words)

        # Find the starting word index of the phrase
        for i in range(idx, len(words) - phrase_len + 1):
            if words[i : i + phrase_len] == phrase_words:
                phrase_locs.append((i, i + phrase_len - 1))
                idx = i + phrase_len
                break

    return phrase_locs


def remove_back_to_back_cr_cl(tokens, cr_token, cl_token):
    """
    Removes `cr_token` followed immediately by `cl_token` from the token list.

    Args:
        tokens (list): The list of tokens.
        cr_token (str): The token indicating the end of a masked phrase.
        cl_token (str): The token indicating the start of a masked phrase.

    Returns:
        list: A new list of tokens with consecutive `cr_token` and `cl_token` removed.
    """
    result = []
    skip_next = False

    for i in range(len(tokens) - 1):
        if skip_next:
            skip_next = False
            continue
        if tokens[i] == cr_token and tokens[i + 1] == cl_token:
            skip_next = True  # Skip the next token
        else:
            result.append(tokens[i])

    # Add the last token if it wasn't skipped
    if not skip_next:
        result.append(tokens[-1])

    return result


# after calling sc for masked_phrases
def insert_sentinel_tokens(input_text, masked_phrases, cl_token, cr_token):
    # Normalize masked phrases: strip extra spaces and remove '�', if any
    masked_phrases = [
        phrase.replace("�", "").strip() for phrase in masked_phrases if phrase.strip()
    ]
    masked_phrases = [
        phrase for phrase in masked_phrases if phrase
    ]  # remove empty strings

    # get the words
    words = input_text.split()

    # find indices where unique masked phrases occur, list of tuples
    masked_phrase_indexes = find_word_level_indexes(words, masked_phrases)

    # for every tuple in masked_phrase_indexes, insert cl_token and cr_token
    phrase_idx_q = deque(masked_phrase_indexes)
    final_tokens = []

    # ensure that there are phrases to mask
    if not phrase_idx_q:
        return input_text

    curr_mask_start, curr_mask_end = phrase_idx_q.popleft()

    for idx, word in enumerate(words):

        if idx == curr_mask_start:
            final_tokens.append(cl_token)

        final_tokens.append(word)

        if idx == curr_mask_end:
            final_tokens.append(cr_token)
            if phrase_idx_q:
                curr_mask_start, curr_mask_end = phrase_idx_q.popleft()

    # combine spans that meet at ends, i.e. remove cr_token, cl_token if they are next to each other
    combined_tokens = remove_back_to_back_cr_cl(final_tokens, cr_token, cl_token)

    return " ".join(combined_tokens)

def strategic_tokenize_function(
    examples: dict,
    compress: bool,
    tokenizer: PreTrainedTokenizer,
    text_column_name: str,
    max_span_length: int,
    bound_ratio: float,
    cl_token: str,
    cr_token: str,
    logger: MultiProcessAdapter,
    sc: SelectiveContext,
) -> dict:
    if not compress:
        # Tokenize and return the column as is
        return {text_column_name: tokenizer(examples[text_column_name])}

    # Prepare the modified column
    modified_texts = []

    for cur_text in examples[text_column_name]:
        # Ignore examples < 30 words
        words = cur_text.split()
        if len(words) < 30:
            tokenized_text = cur_text  # Keep the original text if too short
        else:
            # Use selective context class to determine masked phrases
            _, masked_phrases = sc(
                cur_text, reduce_ratio=bound_ratio, reduce_level="phrase"
            )
            # Insert sentinel tokens
            tokenized_text = insert_sentinel_tokens(
                cur_text, masked_phrases, cl_token, cr_token
            )
        # Append the modified text to the list
        modified_texts.append(tokenized_text)

    # Return only the modified column as a dictionary
    return {text_column_name: modified_texts}
