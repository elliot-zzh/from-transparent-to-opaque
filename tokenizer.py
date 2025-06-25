from transformers import AutoTokenizer

from config import model_name, model_path
from parameters import enable_thinking

prompt = '<|im_start|>user\n Solve the math problem below step by step, and eventually, **repeat your final answer in one single LaTeX `\\boxed{}:`**\n. No need for multi-solution. No need to repeat the solving process in your final answering. Only one `\\boxed` wrapped result is required in your answer. Problem: '  # basic system prompt
prompt_suffix = (
    '/think' if enable_thinking else '/no_think'
) + ' <|im_end|><|im_start|>assistant\n'

tokenizer = (
    AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=model_path)
    if model_path else AutoTokenizer.from_pretrained(model_name, use_fast=True)
)
tokenizer.padding_side = 'left'
