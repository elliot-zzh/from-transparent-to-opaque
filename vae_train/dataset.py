import polars as pl
import torch
import gc
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import model_name
from vae_train.parameters import batch_size, accelerator


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
                # Ensure tensor compatibility before concatenation
                if hasattr(self.data["input_ids"], "_local_tensor") and not hasattr(
                    tmp["input_ids"], "_local_tensor"
                ):
                    tmp["input_ids"] = accelerator.prepare(tmp["input_ids"])
                elif hasattr(tmp["input_ids"], "_local_tensor") and not hasattr(
                    self.data["input_ids"], "_local_tensor"
                ):
                    self.data["input_ids"] = accelerator.prepare(self.data["input_ids"])

                if hasattr(
                    self.data["attention_mask"], "_local_tensor"
                ) and not hasattr(tmp["attention_mask"], "_local_tensor"):
                    tmp["attention_mask"] = accelerator.prepare(tmp["attention_mask"])
                elif hasattr(tmp["attention_mask"], "_local_tensor") and not hasattr(
                    self.data["attention_mask"], "_local_tensor"
                ):
                    self.data["attention_mask"] = accelerator.prepare(
                        self.data["attention_mask"]
                    )

                # Now concatenate with compatible tensor types
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

# Create distributed samplers if using multiple GPUs
train_sampler = (
    DistributedSampler(
        train_data,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
    )
    if accelerator.num_processes > 1
    else None
)

test_sampler = (
    DistributedSampler(
        test_data,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False,
    )
    if accelerator.num_processes > 1
    else None
)

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=(train_sampler is None),
    sampler=train_sampler,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

test_loader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
    sampler=test_sampler,
    num_workers=2,
    pin_memory=True,
)

model.eval()
