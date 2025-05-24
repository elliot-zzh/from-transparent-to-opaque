from datasets import load_dataset
from tqdm import tqdm
from math_dataset import jsonl_dump
import re

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset('BytedTsinghua-SIA/DAPO-Math-17k')

ds = ds['train']

collected = []

for datapoint in tqdm(ds):
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

jsonl_dump(collected, 'data/DAPO-Math-17k/dapo-math-17k.jsonl')
