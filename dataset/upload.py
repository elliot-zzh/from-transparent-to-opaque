from math_dataset import jsonl_read
from datasets import Dataset, DatasetDict

def process_math():
    train = jsonl_read('data/math-train.jsonl')
    test = jsonl_read('data/math-test.jsonl')
    full = train + test
    result = []
    for item in full:
        result.append({
            'question': item['question'],
            'answer': item['answer'],
        })
    length = len(result)
    name = 'math' + str(length // 1000) + 'k'
    return result, name

def process_dapo_math():
    full = jsonl_read('data/dapo-math-17k.jsonl')
    result = []
    for item in full:
        result.append({
            'question': item['question'],
            'answer': item['answer'],
        })
    length = len(result)
    name = 'dapomath' + str(length // 1000) + 'k'
    return result, name

def process_gsm8k():
    train = jsonl_read('data/gsm8k-train.jsonl')
    test = jsonl_read('data/gsm8k-test.jsonl')
    full = train + test
    result = []
    for item in full:
        result.append({
            'question': item['question'],
            'answer': item['answer'],
        })
    length = len(result)
    name = 'gsm' + str(length // 1000) + 'k'
    return result, name

def process_open_r1():
    full = jsonl_read('data/open-r1-math-220k.jsonl')
    result = []
    for item in full:
        result.append({
            'question': item['question'],
            'answer': item['answer'],
        })
    length = len(result)
    name = 'openr1' + str(length // 1000) + 'k'
    return result, name

def process_aime24():
    aime2024 = jsonl_read('data/aime2024i.jsonl') + jsonl_read('data/aime2024ii.jsonl')
    return aime2024

def process_aime25():
    aime2025 = jsonl_read('data/aime2025i.jsonl') + jsonl_read('data/aime2025ii.jsonl')
    return aime2025

def process_gpqa():
    full = jsonl_read('data/gpqa.jsonl')
    return full

def process_math500():
    full = jsonl_read('data/math-500.jsonl')
    return full

def main():
    dataset = {}

    math, math_name = process_math()
    dataset[math_name] = math

    dapo_math, dapo_math_name = process_dapo_math()
    dataset[dapo_math_name] = dapo_math

    gsm8k, gsm8k_name = process_gsm8k()
    dataset[gsm8k_name] = gsm8k

    open_r1, open_r1_name = process_open_r1()
    dataset[open_r1_name] = open_r1

    aime24 = process_aime24()
    dataset['aime24'] = aime24

    aime25 = process_aime25()
    dataset['aime25'] = aime25

    gpqa = process_gpqa()
    dataset['gpqa'] = gpqa

    math500 = process_math500()
    dataset['math500'] = math500

    dsdict = DatasetDict()

    for name, data in dataset.items():
        ds = Dataset.from_list(data)
        dsdict[name] = ds

    dsdict.push_to_hub('ethangoh7086cmd/st-train-all')

if __name__ == '__main__':
    main()