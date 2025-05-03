import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, DynamicCache
from peft import get_peft_model, LoraConfig
from gates import Gate
from vae import VAE
from config import (
    hidden_layer_num,
    depth_start_layer_num,
    hidden_regularization_rate,
    hidden_reg_len_bonus_a,
    hidden_reg_len_bonus_high,
    hidden_dropout_rate,
    eot,
    model_name,
    accelerator,
)
from utils import cleanup


# load the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
torch.backends.cuda.enable_flash_sdp(True)

# inject LoRA
peft_config = LoraConfig(
    init_lora_weights="pissa_niter_4",
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.02,
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Gater
gater = Gate(2048, 0.01)

# load VAE
vae = VAE(2048, 256, 2048 * 4)
vae = torch.jit.script(vae)
vae.load_state_dict(torch.load("/home/featurize/data/vae_epoch15.pth"))


class Model(nn.Module):
    def __init__(self, model, vae, gater):
        super().__init__()
        self.model = model
        self.vae = vae
        self.gater = gater

    def partial_forward(
        hidden_state,
        attn_mask,
        pos,
        kv_cache=None,
        start_layer=0,
        end_layer=model.model.model.config.num_hidden_layers - 1,
        extract_specific=None,
    ):  # hacked
        causal_mask = self.model.model.model._update_causal_mask(
            attention_mask=attn_mask,
            input_tensor=hidden_state,
            cache_position=pos,
            past_key_value=kv_cache,
            output_attentions=False,
        )
        pos_embed = model.model.model.rotary_emb(hidden_state, pos[None, :])

        for layer_i in range(start_layer, end_layer + 1):
            hidden_state = model.model.model.layers[layer_i](
                hidden_state,
                attention_mask=causal_mask,
                position_ids=pos[None, :],
                cache_position=pos,
                past_key_values=kv_cache,
                output_attentions=False,
                use_cache=kv_cache is not None,
                position_embeddings=pos_embed,
            )[0]
            if layer_i == extract_specific:
                specific = hidden_state

            if extract_specific is not None:
                return hidden_state, specific
            return hidden_state

    def forward(
        self,
        input_ids=None,
        input_embeds=None,
        hidden_states=None,
        attn_mask=None,
        training=False,
        lossf=None,
        original_hidden=None,
        pos=None,
        hidden_pos_offset=0,  # should be the length of the questions
        depth=0,
        deep_kv_cache=None,
        kv_cache=None,
        temperature=0.7,
        topk=16,
    ):
        if input_ids is not None and input_embeds is None:
            input_embeds = self.model.model.embed_tokens(input_ids)

        if pos is None:
            pos = torch.arange(input_embeds.shape[1], device=input_embeds.device)

        if hidden_states is not None:
            input_embeds = self.gater(hidden_states, input_embeds)

        B = input_embeds.shape[0]  # batch size

        final_hidden, last_hidden = self.partial_forward(
            hidden_state=input_embeds,
            attn_mask=attn_mask,
            pos=pos,
            kv_cache=kv_cache,
            extract_specific=hidden_layer_num,
        )

        if training:
            logits = self.model.lm_head(
                self.model.model.model.norm(final_hidden)
            ).float()
            loss = lossf()  # TODO: complete loss computation all in here, including gater bonus and regularization
        else:
            logits = self.model.lm_head(
                self.model.model.model.norm(final_hidden[:, -1, :])
            ).float()

        last_hidden_compressed = self.vae(last_hidden, compressing=True)
        last_hidden = self.vae.uncompress(last_hidden_compressed)

        if not training:
            # sampling
            values, indices = torch.topk(
                logits, topk, largest=True, sorted=False, dim=-1
            )
            probs = nn.functional.softmax(values / temperature, dim=-1)
            # probs = probs.masked_fill(probs < min_p * 1 / topk, 0)
            selected_choice = torch.multinomial(probs.view(B, -1), num_samples=1)
            selected_index = indices.gather(1, selected_choice)
            selected_index[(1 - text_end_mask).bool(), :] = eot # FIXME: text_end_mask ?
            res = torch.cat([res, selected_index], dim=1)
            selected_index = selected_index.view(B)

        for depth_i in range(depth):
            last_hidden = self.partial_forward(
                hidden_state=last_hidden,
                attn_mask=attn_mask[:, hidden_pos_offset - 1 :],
                pos=pos[-1:] - hidden_pos_offset + 1,
                kv_cache=deep_kv_cache[depth_i] if not training else None,
                start_layer=depth_start_layer_num,
                end_layer=hidden_layer_num,
            )

        if training:
            return loss, last_hidden_compressed
        else:
            return last_hidden, last_hidden_compressed, selected_index

    def sampler(
        input_ids,
        attn_mask,
        temperature=0.7,
        topk=16,
        max_length=2048,
        gc_interval=64,
        depth=0,
    ):
        self.eval()

        device = input_ids.device
        problem_batch_size = input_ids.shape[0]
        cache_pos = torch.arange(input_ids.shape[1], dtype=torch.long, device=device)
        kv_cache = DynamicCache()
        if depth > 0:
            deep_kv_cache = [DynamicCache() for _ in range(depth)]

        hidden_cache = torch.Tensor(problem_batch_size, 0, 256).to(device)

        # text_end_appeared = False # if the first <｜end▁of▁sentence｜>
        gen_all_done = False

        text_end_mask = torch.ones(problem_batch_size, dtype=torch.int8).to(
            device
        )  # 1 -> not ended
        text_end_indices = torch.ones(problem_batch_size, dtype=torch.long).to(
            device
        ) * (max_length + input_ids.shape[1])

        res = torch.zeros(problem_batch_size, 0, dtype=torch.long).to(device)

        next_input_ids = input_ids
        for i in tqdm(range(max_length), desc="sampling progress"):
            with accelerator.autocast():
                last_hidden, last_hidden_compressed, next_input_ids = self.forward(
                    input_ids=next_input_ids,
                    attn_mask=attn_mask,
                    pos=cache_pos,
                    hidden_states=last_hidden if i > 0 else None,
                    kv_cache=kv_cache,
                    hidden_pos_offset=input_ids.shape[1],
                    temperature=temperature,
                    topk=topk,
                    depth=depth,
                )

                cache_pos = cache_pos[-1:] + 1

                hidden_cache = torch.cat(
                    [
                        hidden_cache,
                        F.dropout(
                            last_hidden_compressed[:, -1:, :],
                            p=hidden_dropout_rate,
                            training=True,
                        ),
                    ],
                    dim=1,
                )

            if i % gc_interval == 0:
                cleanup()

            if not gen_all_done and eot in selected_index:
                # text_end_appeared = True
                text_end_mask.masked_fill_(selected_index == im_end, 0)
                text_end_indices.masked_fill_(selected_index == im_end, i)
                gen_all_done = 1 not in text_end_mask
                # if text_end_mask.sum() < problem_batch_size * 0.2 and text_end_indices.max() + 128 < i:
                #     gen_all_done = True

            attn_mask = torch.cat(
                [attn_mask, text_end_indices.unsqueeze(1)], dim=1
            )  # update attention mask

            if gen_all_done:
                break

        cleanup()

        return res, hidden_cache, text_end_indices, attn_mask
