import re
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import pandas as pd
from math_verify import parse, verify
from tokenizer import prompt, prompt_suffix
from parameters import sample_problem_batch, sample_num
from utils import tokenize
from config import accelerator

data_train = pd.read_json("/home/featurize/data/train.jsonl", lines=True)
data_test = pd.read_json("/home/featurize/data/test.jsonl", lines=True)


class dataset(Dataset):
    def __init__(self, df):
        self.df = df
        print("data tokenizing...")
        self.input_ids, self.attn_mask = tokenize(
            sum(
                [
                    [prompt + df["question"][i] + prompt_suffix] * sample_num
                    for i in df.index
                ],
                [],
            ),
            direct=True,
        )
        self.ans = sum([[df["answer"][i]] * sample_num for i in df.index], [])
        print("data tokenization done")

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, index):
        return self.input_ids[index], self.attn_mask[index], self.ans[index]


train_dataset = dataset(data_train)
test_dataset = dataset(data_test)

# Create samplers for distributed training
train_sampler = (
    DistributedSampler(
        train_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
    )
    if accelerator.num_processes > 1
    else None
)

test_sampler = (
    DistributedSampler(
        test_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False,
    )
    if accelerator.num_processes > 1
    else None
)


data_train = DataLoader(
    train_dataset,
    batch_size=sample_problem_batch * sample_num,
    shuffle=(train_sampler is None),
    sampler=train_sampler,
    num_workers=4,
    pin_memory=True,
)

data_test = DataLoader(
    test_dataset,
    batch_size=sample_problem_batch * sample_num,
    shuffle=False,
    sampler=test_sampler,
    num_workers=4,
    pin_memory=True,
)

data_train, data_test = accelerator.prepare(data_train, data_test)


boxed_match = re.compile(r"\\boxed\{[^}]*\}")


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
