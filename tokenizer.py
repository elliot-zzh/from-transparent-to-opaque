from transformers import AutoTokenizer

from config import model_name, model_path
from parameters import enable_thinking

prompt = "Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering. Problem: "  # basic system prompt
prompt_suffix = '<think>\n'

tokenizer = (
    AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if model_path
    else AutoTokenizer.from_pretrained(model_name, use_fast=True)
)
tokenizer.padding_side = 'left'
