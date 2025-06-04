from datasets import load_dataset
from tqdm import tqdm

from math_dataset import jsonl_dump

def process_gsm8k():
    ds = load_dataset("openai/gsm8k", "main")

    problems_test = []

    for problem in tqdm(ds['test']):
        problems_test.append({
            'question': problem['question'],
            'answer': problem['answer'].rsplit('####', 1)[-1].strip(),
        })

    jsonl_dump(problems_test, 'data/gsm8k-test.jsonl')

    problems_train = []

    for problem in tqdm(ds['train']):
        problems_train.append({
            'question': problem['question'],
            'answer': problem['answer'].rsplit('####', 1)[-1].strip(),
        })

    jsonl_dump(problems_train, 'data/gsm8k-train.jsonl')