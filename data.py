import re
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from math_verify import parse, verify
from model import prompt, prompt_suffix
from parameters import sample_problem_batch, sample_num
from utils import tokenize

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


data_train = DataLoader(
    dataset(data_train), batch_size=sample_problem_batch * sample_num, shuffle=True
)
data_test = DataLoader(dataset(data_test), batch_size=sample_problem_batch * sample_num)


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
