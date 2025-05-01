from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import polars as pl

import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", device_map='auto', torch_dtype=torch.float16, attn_implementation='sdpa')

class VAE(nn.Module):
    def __init__(self, embed_dim, compress_dim, ff_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.compress_dim = compress_dim

        # compressing
        self.norm1 = nn.RMSNorm(embed_dim)
        self.wc1 = nn.Linear(embed_dim, ff_dim)
        self.wcv = nn.Linear(embed_dim, ff_dim)
        self.silu = nn.SiLU()
        self.wc2 = nn.Linear(ff_dim, compress_dim, bias=True)
        self.res_proj1 = nn.Linear(embed_dim, compress_dim, bias=True)

        # uncompressing
        self.norm2 = nn.RMSNorm(compress_dim)
        self.wuc = nn.Linear(compress_dim, ff_dim)
        self.wuv = nn.Linear(compress_dim, ff_dim)
        # self.silu = nn.SiLU()
        self.w_back = nn.Linear(ff_dim, embed_dim)
        self.res_proj2 = nn.Linear(compress_dim, embed_dim)

    def uncompress(self, x):
        x = self.norm2(x)
        return self.w_back(self.silu(self.wuc(x)) * self.wuv(x)) + self.res_proj2(x)

    def forward(self, x, compressing=False):
        x = self.norm1(x)
        x = self.wc2(self.silu(self.wc1(x)) * self.wcv(x)) + self.res_proj1(x)
        if compressing: return x
        return self.uncompress(x)

vae = VAE(2048, 256, 2048 * 4)

class Data(Dataset):
    def __init__(self, data_raw):
        super().__init__()
        if len(data_raw) <= 10000:
            self.data = tokenizer(data_raw.tolist(), return_tensors='pt', truncation=True, padding='max_length', max_length=1536)
        else:
            data = data_raw[:len(data_raw) - len(data_raw) % 10000].reshape(-1, 10000).tolist() + [data_raw[len(data_raw) - len(data_raw) % 10000:].tolist()]
            self.data = tokenizer(data[0], return_tensors='pt', truncation=True, padding='max_length', max_length=1536)
            for i in data[1:]:
                gc.collect()
                tmp = tokenizer(i, return_tensors='pt', truncation=True, padding='max_length', max_length=1536)
                self.data['input_ids'] = torch.cat([self.data['input_ids'], tmp['input_ids']], dim=0)
                self.data['attention_mask'] = torch.cat([self.data['attention_mask'], tmp['attention_mask']], dim=0)
            
    def __getitem__(self, index):
        return self.data['input_ids'][index], self.data['attention_mask'][index]

    def __len__(self):
        return self.data['input_ids'].shape[0]

batch_size = 20
data_raw = pl.read_parquet('/home/featurize/data/res-sampled.parquet')['text'].to_numpy()
total_size = len(data_raw)
train_data = Data(data_raw[:int(0.96 * total_size)])
test_data = Data(data_raw[int(0.96 * total_size):total_size])
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=2, pin_memory=True)
print(total_size)

optimizer = Adam(vae.parameters(), lr=1e-3)

num_epochs = 30
gradient_accumulation_steps = 6
log_interval = 1
hidden_layer_num = 20
scaler = torch.amp.GradScaler(device=device)
lossf = nn.MSELoss()

model.eval()
vae = vae.to(device)
vae.train()

def cleanup():
    gc.collect()
    if device == 'cuda': torch.cuda.empty_cache()
    elif device == 'mps': torch.mps.empty_cache()

for epoch in range(num_epochs):
    for step, (input_ids, attn_mask) in enumerate(train_loader):
        try:
            cleanup()
            
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=str(device), dtype=torch.float16):
                with torch.no_grad():
                    hidden = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True, return_dict=True).hidden_states[hidden_layer_num].float()
            loss = lossf(vae(hidden), hidden)
            loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if (step + 1) % (gradient_accumulation_steps * log_interval) == 0:
                print(f"Epoch {epoch+1}, Step {step+1}, Train Loss: {loss.item():.3f}")

        except KeyboardInterrupt:
            cleanup()

    if (step + 1) % gradient_accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        print(f"Epoch {epoch+1}, Step {step+1}, Train Loss: {loss.item():.3f}")

    count = 0
    loss = 0
    vae.eval()
    for step, (input_ids, attn_mask) in enumerate(test_loader):
        try:
            cleanup()
            
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)

            with torch.no_grad():
                with torch.amp.autocast(device_type=str(device), dtype=torch.float16):
                    hidden = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True, return_dict=True).hidden_states[hidden_layer_num].float()
                loss += lossf(vae(hidden), hidden)
                count += 1

        except KeyboardInterrupt:
            cleanup()

    print(f"Epoch {epoch+1}, Test Loss: {loss.item() / count:.3f}")
         
    # Save checkpoint
    torch.save(vae.state_dict(), f"vae_epoch{epoch+1}.pth")