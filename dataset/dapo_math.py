from datasets import load_dataset
from tqdm import tqdm
from math_dataset import jsonl_dump
from transformers import AutoTokenizer
import re


def process_dapo_math():
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B')

    ds = load_dataset('BytedTsinghua-SIA/DAPO-Math-17k')

    ds = ds['train']

    collected = []

    for datapoint in tqdm(ds):
        question = datapoint['prompt'][0]['content'].lstrip(
                    'Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n'
                ).rstrip('\n\nRemember to put your answer on its own line after "Answer:".')
        question_tokens = tokenizer(question, return_tensors='pt')
        markdown_image_pattern = re.compile(r'!\[.*?\]\(.*?\)')
        full_ascii_checker = re.compile(r'^[\x00-\x7F]+$')
        if not full_ascii_checker.match(question):
            continue
        if question_tokens.input_ids.shape[1] > 256:
            continue
        if markdown_image_pattern.search(question):
            continue
        collected.append(
            {
                'question': datapoint['prompt'][0]['content']
                .lstrip(
                    'Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n'
                )
                .rstrip('\n\nRemember to put your answer on its own line after "Answer:".'),
                'answer': datapoint['reward_model']['ground_truth'],
            }
        )

    jsonl_dump(collected, 'data/dapo-math-17k.jsonl')

if __name__ == '__main__':
    process_dapo_math()