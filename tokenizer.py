from transformers import AutoTokenizer
from config import model_name


tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.padding_side = "left"
