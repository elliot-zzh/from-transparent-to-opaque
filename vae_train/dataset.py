import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import model_name
from vae_train.parameters import batch_size

import gc


class Data(Dataset):
    def __init__(self, data_raw):
        super().__init__()
        if len(data_raw) <= 10000:
            self.data = tokenizer(
                data_raw.tolist(),
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=1536,
            )
        else:
            data = data_raw[: len(data_raw) - len(data_raw) % 10000].reshape(
                -1, 10000
            ).tolist() + [data_raw[len(data_raw) - len(data_raw) % 10000 :].tolist()]
            self.data = tokenizer(
                data[0],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=1536,
            )
            for i in data[1:]:
                gc.collect()
                tmp = tokenizer(
                    i,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=1536,
                )
                self.data["input_ids"] = torch.cat(
                    [self.data["input_ids"], tmp["input_ids"]], dim=0
                )
                self.data["attention_mask"] = torch.cat(
                    [self.data["attention_mask"], tmp["attention_mask"]], dim=0
                )

    def __getitem__(self, index):
        return self.data["input_ids"][index], self.data["attention_mask"][index]

    def __len__(self):
        return self.data["input_ids"].shape[0]


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype=torch.float16, attn_implementation="sdpa"
)

data_raw = pl.read_parquet("/home/featurize/data/res-sampled.parquet")[
    "text"
].to_numpy()
total_size = len(data_raw)
train_data = Data(data_raw[: int(0.96 * total_size)])
test_data = Data(data_raw[int(0.96 * total_size) : total_size])
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
test_loader = DataLoader(
    test_data, batch_size=batch_size, num_workers=2, pin_memory=True
)

model.eval()
