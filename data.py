import re
from datasets import load_dataset
from math_verify import parse, verify
from torch.utils.data import DataLoader, Dataset
from parameters import (
    sample_num,
    sample_problem_batch,
    hf_dataset_name,
    hf_train_split,
    enable_swapping,
)
from tokenizer import prompt, prompt_suffix

# Only load the specified split
train_data = load_dataset(hf_dataset_name, split=hf_train_split)


class dataset(Dataset):
    def __init__(self, hf_split):
        self.hf_split = hf_split
        self.sample_num = sample_num
        self.enable_swapping = enable_swapping
        self.prompt = prompt
        self.prompt_suffix = prompt_suffix

    def __len__(self):
        base_len = len(self.hf_split)
        return base_len * (self.sample_num if self.enable_swapping else 1)

    def __getitem__(self, index):
        base_len = len(self.hf_split)
        question_index = index // self.sample_num if self.enable_swapping else index
        item = self.hf_split[question_index]
        problem = self.prompt + item['question'] + self.prompt_suffix
        answer = item['answer']
        return problem, answer


data_train = DataLoader(
    dataset(train_data), batch_size=sample_problem_batch * sample_num, shuffle=True
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
