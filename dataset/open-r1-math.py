from datasets import load_dataset
from tqdm import tqdm
from math_dataset import jsonl_dump
import re

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("open-r1/OpenR1-Math-220k", "default")

ds = ds['train']


markdown_image_pattern = re.compile(r'!\[.*?\]\(.*?\)')
text = "This is a paragraph with an image ![example](https://example.com/image.png)"


collected = []

for datapoint in tqdm(ds):
    if datapoint['question_type'] != 'MCQ' and not markdown_image_pattern.search(datapoint['problem']):
        collected.append({
            'question': datapoint['problem'],
            'answer': datapoint['answer'].replace('\\mathrm{~}', '').strip(),
            'source': datapoint['source']
        })

jsonl_dump(collected, 'data/OpenR1-Math-220k/open-r1-math-220k.jsonl')
