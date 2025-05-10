from transformers import AutoTokenizer
from config import model_name

prompt = "<|im_start|>/no_think Solve the math problem below, step by step in detail, and eventually, **repeat your final answer in the LaTeX `\\boxed{}:`**\n"  # basic system prompt
prompt_suffix = "\n<|im_end|><|im_start|>\n"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.padding_side = "left"
