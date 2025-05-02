import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DynamicCache,
    OffloadedCache,
)
import datasets
from peft import get_peft_model, LoraConfig

from tqdm import tqdm

import gc
import re

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

import pandas as pd
import matplotlib.pyplot as plt

from math_verify import parse, verify

torch.manual_seed(42)

writer = SummaryWriter("runs/demo")

# load the model
model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype=torch.float16, attn_implementation="sdpa"
)
torch.backends.cuda.enable_flash_sdp(True)


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

    def forward(self, x, compressing: bool = False):
        x = self.norm1(x)
        x = self.wc2(self.silu(self.wc1(x)) * self.wcv(x)) + self.res_proj1(x)
        if compressing:
            return x
        return self.uncompress(x)


class CurrentStepMixerGater(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.w = nn.Linear(embed_dim * 2, embed_dim)
        # self.act = nn.Tanh()
        # self.act = nn.SiLU()
        self.act = nn.Sigmoid()
        # torch.nn.init.constant_(self.w.weight, 1 / embed_dim)
        torch.nn.init.zeros_(self.w.weight)

    def forward(self, hidden, embed):
        x = torch.cat([hidden, embed], dim=-1)
        return self.act(self.w(x))


class Gate(nn.Module):
    def __init__(self, embed_dim, inject_scale, zero_init=True, dropout_rate=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.inject_scale = inject_scale

        self.norm = nn.RMSNorm(embed_dim)
        if zero_init:
            self.gate = nn.Parameter(
                torch.zeros(embed_dim)
            )  # all from model embeddings first for stability
        else:
            self.gate = nn.Parameter(torch.ones(embed_dim) * 0.5)
        self.time_mixing_gate = CurrentStepMixerGater(embed_dim)

    def forward(self, hidden, embed):
        hidden = self.norm(hidden)
        return embed * (
            1 - self.gate
        ) + self.inject_scale * self.gate * hidden * self.time_mixing_gate(
            hidden, embed
        )

    def forward_hidden(self, hidden, embed):  # forward hidden only
        return self.gate * hidden * self.time_mixing_gate(hidden, embed)

    def print_gates(self):
        print("gate value:", self.gate[:20])

    def print_heatmap(self):
        plt.imshow(
            self.gate.detach().cpu().numpy()[:20], cmap="hot", interpolation="nearest"
        )
        plt.colorbar()
        plt.show()


# inject LoRA
peft_config = LoraConfig(
    init_lora_weights="pissa_niter_4",
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    target_modules="all-linear",  # ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
    lora_dropout=0.05,
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Gater
gater = Gate(2048, 0.01)

# load VAE
vae = VAE(2048, 256, 2048 * 4)
vae = torch.jit.script(vae)
vae.load_state_dict(torch.load("/home/featurize/data/vae_epoch15.pth"))

vae = vae.to(device)
gater = gater.to(device)

# end_of_text mark
# eot = tokenizer('<｜end▁of▁sentence｜>').input_ids[1:][0]
im_end, eot = tokenizer("<|im_end|><|endoftext|>").input_ids

hidden_layer_num = 20
depth_start_layer_num = 12


def cleanup():
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.cuda.empty_cache()


def tokenize(text, direct=False, max_length=1024, pad=False, device=device):
    if direct:
        res = tokenizer(text, return_tensors="pt", padding=True)
    else:
        res = tokenizer(
            text,
            return_tensors="pt",
            truncation=trn,
            max_length=max_length,
            padding="max_length",
        )
    input_ids = res.input_ids.to(device)
    attn_mask = res.attention_mask.to(device)
    return input_ids, attn_mask


# A lot of hacking here. For details please refer to
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/
# and
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/
def sampler(
    input_ids,
    attn_mask,
    temperature=0.7,
    topk=16,
    max_length=2048,
    num=16,
    min_p=0.02,
    gc_interval=64,
    hidden_dropout_rate=0.02,
    depth=0,
):
    model.eval()
    vae.eval()
    gater.eval()

    # tokenize
    problem_batch_size = input_ids.shape[0]
    cache_pos = torch.arange(input_ids.shape[1], dtype=torch.long).to(device)
    kv_cache = DynamicCache()
    if depth > 0:
        deep_kv_cache = [DynamicCache() for _ in range(depth)]

    # prefill the problem
    with torch.amp.autocast(device_type=str(device), dtype=torch.float16):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
            return_dict=True,
            cache_position=cache_pos,
            past_key_values=kv_cache,
        )

    last_hidden = outputs.hidden_states[hidden_layer_num]
    hidden_cache = torch.Tensor(problem_batch_size, 0, 256).to(device)

    # text_end_appeared = False # if the first <｜end▁of▁sentence｜>
    gen_all_done = False

    text_end_mask = torch.ones(problem_batch_size, dtype=torch.int8).to(
        device
    )  # 1 -> not ended
    text_end_indices = torch.ones(problem_batch_size, dtype=torch.long).to(device) * (
        max_length + input_ids.shape[1]
    )

    res = torch.zeros(problem_batch_size, 0, dtype=torch.long).to(device)

    hidden_stream = (
        torch.cuda.Stream()
    )  # an extra cuda stream for hidden vector processing

    for i in tqdm(range(max_length), desc="sampling progress"):
        with torch.cuda.stream(hidden_stream):
            with torch.amp.autocast(device_type=str(device), dtype=torch.float16):
                last_hidden = outputs.hidden_states[hidden_layer_num]
                hidden_cache = torch.cat(
                    [
                        hidden_cache,
                        F.dropout(
                            vae(last_hidden[:, -1:, :], compressing=True),
                            p=hidden_dropout_rate,
                            training=True,
                        ),
                    ],
                    dim=1,
                )
                uncompressed_hidden = vae.uncompress(hidden_cache[:, -1:, :])

                # more depth -- model looping
                if depth > 0:
                    last_hidden = uncompressed_hidden
                    deep_cache_pos = cache_pos[-1:] - input_ids.shape[1] + 1
                    causal_mask = model.model.model._update_causal_mask(
                        attention_mask=attn_mask[
                            :, input_ids.shape[1] + i - 1
                        ].unsqueeze(1),
                        input_tensor=last_hidden,
                        cache_position=deep_cache_pos,
                        past_key_values=deep_kv_cache[0],
                        output_attentions=False,
                    )
                    pos_embed = model.model.model.rotary_emb(
                        last_hidden, deep_cache_pos.unsqueeze(0)
                    )
                    for depth_i in range(depth):
                        for layer_i in range(depth_start_layer_num, hidden_layer_num):
                            last_hidden = model.model.model.layers[layer_i](
                                last_hidden,
                                attention_mask=causal_mask,
                                position_ids=deep_cache_pos,
                                cache_position=deep_cache_pos,
                                past_key_values=deep_kv_cache[depth_i],
                                output_attentions=False,
                                use_cache=True,
                                position_embeddings=pos_embed,
                            )[0]
                    uncompressed_hidden = last_hidden

        logits = outputs.logits[:, -1, :].float()  # (problem_batch_size, vocab_size)

        if i % gc_interval == 0:
            cleanup()

        values, indices = torch.topk(logits, topk, largest=True, sorted=False, dim=-1)
        probs = nn.functional.softmax(values / temperature, dim=-1)
        # probs = probs.masked_fill(probs < min_p * 1 / topk, 0)
        selected_choice = torch.multinomial(
            probs.view(problem_batch_size, -1), num_samples=1
        )
        selected_index = indices.gather(1, selected_choice)
        selected_index[(1 - text_end_mask).bool(), :] = eot
        res = torch.cat([res, selected_index], dim=1)
        selected_index = selected_index.view(problem_batch_size)

        if not gen_all_done and eot in selected_index:
            text_end_appeared = True
            text_end_mask.masked_fill_(selected_index == im_end, 0)
            text_end_indices.masked_fill_(selected_index == im_end, i)
            gen_all_done = not (1 in text_end_mask)
            # if text_end_mask.sum() < problem_batch_size * 0.2 and text_end_indices.max() + 128 < i:
            #     gen_all_done = True

        attn_mask = torch.cat(
            [attn_mask, text_end_indices.unsqueeze(1)], dim=1
        )  # update attention mask

        if gen_all_done:
            break

        # forward
        with torch.amp.autocast(device_type=str(device), dtype=torch.float16):
            cache_pos = cache_pos[-1:] + 1
            embeds = model.lm_head.weight[
                selected_index.view(problem_batch_size, 1)
            ].to(device)
            torch.cuda.current_stream().wait_stream(hidden_stream)
            embeds = gater(uncompressed_hidden, embeds)
            outputs = model(
                inputs_embeds=embeds,
                attention_mask=attn_mask,
                output_hidden_states=True,
                cache_position=cache_pos,
                return_dict=True,
                use_cache=True,
                past_key_values=kv_cache,
            )

    cleanup()
    if depth > 0:
        return res, hidden_cache, text_end_indices, attn_mask
    return res, hidden_cache, text_end_indices, attn_mask


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


prompt = "<|im_start|>Human: You are a math solving assistant. Now you should solve the math problem below, step by step in detail, and eventually, **reqeat your final answer in the latex `\\boxed{}:`**\n"  # basic system prompt
prompt_suffix = "\n<|im_end|><|im_start|>\n"

# load dataset
# from datasets import load_dataset
# data = load_dataset('open-r1/OpenR1-Math-220k', split='train')
data_train = pd.read_json("/home/featurize/data/train.jsonl", lines=True)
data_test = pd.read_json("/home/featurize/data/test.jsonl", lines=True)
from torch.utils.data import Dataset


class dataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, index):
        return self.df["question"][index], self.df["answer"][index]


optimizers = [
    AdamW(model.parameters(), lr=1e-5),
    AdamW(vae.parameters(), lr=5e-5),
    Adam(gater.parameters(), lr=3e-3),
]
gater_scheduler = CosineAnnealingLR(optimizers[2], T_max=500, eta_min=1e-3)
scaler = torch.amp.GradScaler(device=device)
lossf = nn.CrossEntropyLoss(reduction="none")
hidden_regularizer = nn.MSELoss(reduction="none")


def save_model(steps):
    model.save_pretrained(f"./model-{steps}")
    torch.save(vae.state_dict(), f"vae-{steps}.pt")
    torch.save(gater.state_dict(), f"gater-{steps}.pt")


def step_optimizer():
    for optim in optimizers:
        scaler.step(optim)
    scaler.update()
    gater_scheduler.step()


def zero_grad_optimizer():
    for optim in optimizers:
        optim.zero_grad(set_to_none=True)


num_epochs = 4  # for each RL batch
total_steps = 1000  # on the whole data
log_interval = 1
save_interval = 1024
batch_size = 1
max_train_length = 1024
max_sample_length = 512
l_cache_length = 400
sample_num = 16
sample_topk = 12
sample_temperature = 0.6
sample_problem_batch = 5
sample_problem_sub_batch = 5
acc_check_only = False
train_gc_interval = 15
corr_reward = 2

# hidden regularization
hidden_regularization_rate = 0.5
hidden_dropout_rate = 0.05
hidden_reg_len_bonus_a = 20
hidden_reg_len_bonus_high = 10
hidden_updating_rate = 0.05

# gating value bonus
gating_value_bonus = 0.2
gating_value_decay = 0.95
gating_value_lambda = 5
gating_bonus_update_step = 100

looping_depth = 1  # not ready for depth > 0 yet

data_train = DataLoader(
    dataset(data_train), batch_size=sample_problem_batch, shuffle=True
)
data_test = DataLoader(dataset(data_test), batch_size=sample_problem_batch)

accumulated_step = 1
step = 1  # total step count


def linear_interpl(
    x: torch.Tensor, a: float, b: float, low: float, high: float
) -> torch.Tensor:  # only interpl between [a, b], linearly increase from low to high
    mask_low = x <= a
    mask_high = x >= b
    mask_mid = ~mask_low & ~mask_high

    t = (x - a) / (b - a)
    mid_values = low + (high - low) * t

    return torch.where(mask_low, low, torch.where(mask_high, high, mid_values))


def norm(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean()
    return x / (x**2).mean() * 0.5


linear_interpl = torch.jit.script(linear_interpl)
norm = torch.jit.script(norm)

while step <= total_steps:
    for problem, ans in data_train:
        ans = [re.search(r"####\s*(.*)$", i).group(1) for i in ans]
        cleanup()
        with torch.no_grad():
            init_res = False
            # multi turn sampling
            ans = sum([[i] * sample_num for i in ans], [])
            input_ids, problem_attn_mask = tokenize(
                sum([[prompt + i + prompt_suffix] * sample_num for i in problem], []),
                direct=True,
            )
            for i in range(0, sample_problem_batch, sample_problem_sub_batch):
                if init_res:
                    res_, hidden_cache_, text_end_indices_, mask_ = sampler(
                        input_ids[
                            i * sample_num : (i + sample_problem_sub_batch) * sample_num
                        ],
                        problem_attn_mask[
                            i * sample_num : (i + sample_problem_sub_batch) * sample_num
                        ],
                        num=sample_num,
                        topk=sample_topk,
                        max_length=max_sample_length,
                        temperature=sample_temperature,
                        depth=looping_depth,
                    )
                    res = torch.cat([res, res_], dim=0)
                    hidden_cache = torch.cat([hidden_cache, hidden_cache_], dim=0)
                    text_end_indices = torch.cat(
                        [text_end_indices, text_end_indices_], dim=0
                    )
                    mask = torch.cat([mask, mask_], dim=0)
                else:
                    res, hidden_cache, text_end_indices, mask = sampler(
                        input_ids[: sample_problem_sub_batch * sample_num],
                        problem_attn_mask[: sample_problem_sub_batch * sample_num],
                        num=sample_num,
                        topk=sample_topk,
                        max_length=max_sample_length,
                        depth=looping_depth,
                    )
                    init_res = True

                cleanup()

            cleanup()
            hidden_cache = hidden_cache[:, :-1]

            correctness_rewards = torch.Tensor(
                verifier(
                    tokenizer.batch_decode(res, skip_special_tokens=True),
                    ans,
                    corr_score=corr_reward,
                )
            ).to(device)
            len_rewards = text_end_indices.float() + 1

            # normalization
            filt = None
            if (
                l := (corr_filt := correctness_rewards == corr_reward).sum()
            ) < res.shape[
                0
            ] / 3 and l != 0:  # clip too many wrong answers, currently 1:1
                incorr_filt = torch.ones(sample_num * sample_problem_batch).to(device)
                incorr_filt[correctness_rewards == 1] = 0
                incorr_filt = torch.multinomial(incorr_filt, num_samples=l * 2)
                filt = torch.cat(
                    [torch.nonzero(corr_filt, as_tuple=True)[0], incorr_filt], dim=0
                )
                filt = filt[torch.randperm(filt.size(0))]
                correctness_rewards = correctness_rewards[filt]
                len_rewards = len_rewards[filt]
                mask = mask[filt]
                input_ids = input_ids[filt]
                hidden_cache = hidden_cache[filt]
                res = res[filt]
                text_end_indices = text_end_indices[filt]

            correctness = l.cpu().item()
            print("correctness: ", correctness)
            writer.add_scalar("correctness/train", correctness, step)

            max_len_mask = len_rewards >= max_sample_length
            if max_len_mask.any():
                len_rewards[len_rewards >= max_sample_length] = -1
            cache_len_mask = len_rewards <= l_cache_length
            if cache_len_mask.any():
                len_rewards[len_rewards <= l_cache_length] = 0
            len_interval_mask = torch.logical_not(
                torch.logical_or(max_len_mask, cache_len_mask)
            )
            if len_interval_mask.any():
                len_rewards[len_interval_mask] = (
                    l_cache_length - len_rewards[len_interval_mask]
                ) / (max_sample_length - l_cache_length)
            rewards = correctness_rewards + len_rewards
            rewards = norm(rewards)

        if acc_check_only:
            continue

        with torch.no_grad():
            if res.shape[1] > max_train_length:
                seqs = torch.cat([input_ids, res[:, :max_train_length]], dim=1)
                hidden_cache = hidden_cache[:, : max_train_length - 1]
            else:
                seqs = torch.cat([input_ids, res], dim=1)

        # training
        print("start training")
        model.train()
        vae.train()
        gater.train()
        for epoch in range(num_epochs):
            print("training epoch: ", epoch + 1)
            cleanup()
            new_hidden_cache = torch.zeros_like(hidden_cache)
            for i in range(0, res.shape[0], batch_size):
                try:
                    if step % train_gc_interval == 0:
                        cleanup()
                    end = (
                        (i + batch_size)
                        if i + batch_size <= res.shape[0]
                        else res.shape[0]
                    )
                    embeds = model.lm_head.weight[seqs[i:end]][:, :-1].to("cuda:0")
                    hidden_cache_slice = hidden_cache[i:end]
                    with torch.amp.autocast(
                        device_type=str(device), dtype=torch.float16
                    ):
                        last_hidden = vae.uncompress(
                            F.dropout(
                                hidden_cache_slice, p=hidden_dropout_rate, training=True
                            )
                        )
                        if looping_depth > 0:  # deep looping
                            hidden_pos = torch.arange(
                                0, last_hidden.shape[1], dtype=torch.long, device=device
                            )
                            causal_mask = model.model.model._update_causal_mask(
                                attention_mask=mask[i:end, input_ids.shape[1] - 1 : -1],
                                input_tensor=last_hidden,
                                cache_position=hidden_pos,
                                output_attentions=False,
                                past_key_values=None,
                            )  # the mask here needs to be re-considered
                            pos_embed = model.model.model.rotary_emb(
                                last_hidden, hidden_pos.unsqueeze(0)
                            )
                            for depth_i in range(looping_depth):
                                for layer_i in range(
                                    depth_start_layer_num, hidden_layer_num
                                ):
                                    last_hidden = model.model.model.layers[layer_i](
                                        last_hidden,
                                        attention_mask=causal_mask,
                                        position_ids=hidden_pos,
                                        output_attentions=False,
                                        use_cache=False,
                                        cache_position=hidden_pos,
                                        position_embeddings=pos_embed,
                                    )[0]
                        embeds = torch.cat(
                            [
                                embeds[:, : input_ids.shape[1]],
                                gater(last_hidden, embeds[:, input_ids.shape[1] :]),
                            ],
                            dim=1,
                        )
                        outputs = model(
                            inputs_embeds=embeds,
                            attention_mask=mask[i:end],
                            output_hidden_states=True,
                            return_dict=True,
                        )
                        loss = lossf(
                            outputs.logits[:, input_ids.shape[1] - 1 :].transpose(1, 2),
                            seqs[i:end, input_ids.shape[1] :].masked_fill(
                                mask[i:end, input_ids.shape[1] :] == 0, -100
                            ),
                        )
                        hidden = outputs.hidden_states[hidden_layer_num]

                        # compute loss
                        new_compressed_hidden = vae(
                            hidden[:, input_ids.shape[1] - 1 : -1], compressing=True
                        )
                        new_processed_hidden = vae.uncompress(new_compressed_hidden)
                        if looping_depth > 0:  # deep looping
                            hidden_pos = torch.arange(
                                0,
                                new_processed_hidden.shape[1],
                                dtype=torch.long,
                                device=device,
                            )
                            causal_mask = model.model.model._update_causal_mask(
                                attention_mask=mask[i:end, input_ids.shape[1] - 1 : -1],
                                cache_position=hidden_pos,
                                input_tensor=new_processed_hidden,
                                output_attentions=False,
                                past_key_values=None,
                            )  # the mask here needs to be re-considered
                            pos_embed = model.model.model.rotary_emb(
                                new_processed_hidden, hidden_pos.unsqueeze(0)
                            )
                            for depth_i in range(looping_depth):
                                for layer_i in range(
                                    depth_start_layer_num, hidden_layer_num
                                ):
                                    new_processed_hidden = model.model.model.layers[
                                        layer_i
                                    ](
                                        last_hidden,
                                        attention_mask=causal_mask,
                                        position_ids=hidden_pos,
                                        output_attentions=False,
                                        use_cache=False,
                                        cache_position=hidden_pos,
                                        position_embeddings=pos_embed,
                                    )[0]
                        new_processed_hidden = gater.forward_hidden(
                            new_processed_hidden, embeds[:, input_ids.shape[1] :]
                        )
                        processed_hidden = gater.forward_hidden(
                            last_hidden, embeds[:, input_ids.shape[1] :]
                        )
                        hidden_loss = (
                            hidden_regularizer(
                                new_processed_hidden, processed_hidden
                            ).mean(dim=-1)
                            * mask[i:end, input_ids.shape[1] : -1]
                        )
                        # apply hidden regularization bonus
                        hidden_loss = hidden_loss * linear_interpl(
                            (text_end_indices + 1)[i : i + batch_size],
                            hidden_reg_len_bonus_a,
                            max_sample_length,
                            1,
                            hidden_reg_len_bonus_high,
                        )
                        # apply gating value bonus
                        gate_bonus = torch.exp(
                            gating_value_lambda * (0.5 - gater.gate) ** 2
                        ).mean()
                        loss = (
                            (loss.sum(dim=-1) * rewards[i:end]).sum()
                            + hidden_loss.sum() * hidden_regularization_rate
                        ) / (text_end_indices[i : i + batch_size] + 1).sum()
                        loss += (
                            gate_bonus
                            * gating_value_bonus
                            * gating_value_decay ** (step // gating_bonus_update_step)
                        )
                        loss *= batch_size / res.shape[0]

                        # randomly update hidden cache
                        with torch.no_grad():
                            update_index = torch.nonzero(
                                torch.randn(hidden_cache.shape[1], device=device)
                                < hidden_updating_rate
                            )
                            hidden_cache[i:end][:, update_index] = (
                                new_compressed_hidden[:, update_index]
                            )

                    scaler.scale(loss).backward()

                except KeyboardInterrupt:
                    cleanup()
                    exit()

            step_optimizer()
            zero_grad_optimizer()

            print(f"Step {step}, Loss: {loss.item():.3f}")
            gater.print_gates()

            step += 1

            cleanup()

    # Save checkpoint
    if step % save_interval == 0:
        save_model(step)

print("all done")
