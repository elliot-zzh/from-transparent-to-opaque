from transformers import AutoTokenizer
from config import model_name
from parameters import enable_thinking

prompt = '<|im_start|>user\n You are in a mathematics test. Solve the math problem below step by step, and eventually, repeat your final answer in the `\\boxed{}:`\n. No need to overthink, no need for multi-solution. If verification is needed (optional), first repeat your answer via `\\boxed{}` before verification, because only your last repeated answer will be checked. Problem here: '  # basic system prompt
prompt_suffix = (
    '/think' if enable_thinking else '/no_think'
) + ' <|im_end|><|im_start|>assistant\n'

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.padding_side = 'left'
