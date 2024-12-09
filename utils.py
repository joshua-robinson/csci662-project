from transformers import AutoConfig


def ckpt_associated_with_gptneox(ckpt: str) -> bool:
    arch_str = AutoConfig.from_pretrained(ckpt).architectures[0]
    return arch_str == "GPTNeoXForCausalLM"
