from datasets import load_dataset
from tqdm import tqdm
from math_dataset import jsonl_dump

def process_gpqa():
    ds = load_dataset("fingertap/GPQA-Diamond")

    problems = []

    for problem in tqdm(ds['test']):
        problems.append({
            'question': problem['question'] + '\n You answer should be only a capitalized letter, e.g. A, B, C, D.',
            'answer': problem['answer'].upper(),
        })

    jsonl_dump(problems, 'data/gpqa.jsonl')

if __name__ == '__main__':
    process_gpqa()
    print('Done!')