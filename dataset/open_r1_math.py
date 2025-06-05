from datasets import load_dataset
from math_dataset import jsonl_dump
import re
from transformers import AutoTokenizer

def process_open_r1_math():
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B')

    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset('open-r1/OpenR1-Math-220k', 'default')

    ds = ds['train']

    markdown_image_pattern = re.compile(r'!\[.*?\]\(.*?\)')
    text = 'This is a paragraph with an image ![example](https://example.com/image.png)'

    collected = []

    full_ascii_checker = re.compile(r'^[\x00-\x7F]+$')

    for idx, datapoint in enumerate(ds):
        if datapoint['question_type'] != 'MCQ' and not markdown_image_pattern.search(
                datapoint['problem']
        ):
            problem_tokens = tokenizer(
                datapoint['problem'], return_tensors='pt'
            )
            if not full_ascii_checker.match(datapoint['problem']):
                continue
            if problem_tokens.input_ids.shape[1] > 256:
                continue
            collected.append(
                {
                    'question': datapoint['problem'],
                    'answer': datapoint['answer'].replace('\\mathrm{~}', '').strip(),
                    'source': datapoint['source'],
                }
            )
        if idx % 1024 == 1023:
            print(f'Processed {idx} samples...')

    jsonl_dump(collected, 'data/open-r1-math-220k.jsonl')
