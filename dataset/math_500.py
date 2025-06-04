from datasets import load_dataset
from tqdm import tqdm
from math_dataset import jsonl_dump

def process_math_500():
    ds = load_dataset("HuggingFaceH4/MATH-500")

    problems = []

    for problem in tqdm(ds['test']):
        problems.append({
            'question': problem['problem'],
            'answer': problem['answer'],
        })

    jsonl_dump(problems, 'data/math-500.jsonl')
