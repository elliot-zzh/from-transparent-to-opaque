import re
import polars as pl
import json
from math_verify import parse, verify
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from parameters import (
    sample_num,
    sample_problem_batch,
    test_dataset_path,
    train_dataset_path,
)
from tokenizer import prompt, prompt_suffix

def read_jsonl_with_progress(file_path):
    data = []
    with open(file_path, 'r') as f:
        total_lines = sum(1 for _ in f)
    with open(file_path, 'r') as f:
        for line in tqdm(f, total=total_lines, desc=f"Reading {file_path}"):
            data.append(json.loads(line.strip()))
    return pl.DataFrame(data)

data_train = read_jsonl_with_progress(train_dataset_path)
data_test = read_jsonl_with_progress(test_dataset_path)

class dataset(Dataset):
    def __init__(self, df):
        questions = df["question"].to_list()
        answers = df["answer"].to_list()
        self.problems = [prompt + question + prompt_suffix for question in questions]
        self.answers = answers
        self.sample_num = sample_num

    def __len__(self):
        return len(self.problems) * self.sample_num

    def __getitem__(self, index):
        question_index = index // self.sample_num
        return self.problems[question_index], self.answers[question_index]

data_train = DataLoader(
    dataset(data_train),
    batch_size=sample_problem_batch * sample_num,
    shuffle=True
)

boxed_match = re.compile(r'\\boxed\{[^}]*\}')

def verifier(model_anss, corr_anss, corr_score=2, wrong_score=-1):
    res = []
    for idx, i in enumerate(model_anss):
        model_ans = boxed_match.findall(i)
        if model_ans:
            model_ans = parse(model_ans[-1])
            res.append(
                corr_score
                if verify(model_ans, parse(corr_anss[idx % len(corr_anss)]))
                else wrong_score
            )
        else:
            res.append(wrong_score)
    return res
