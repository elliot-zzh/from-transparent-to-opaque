from model import model
import torch


def model_forward(
    hidden_state,
    attn_mask,
    pos,
    kv_cache=None,
):  # hacked
    causal_mask = model.model.model._update_causal_mask(
        attention_mask=attn_mask,
        input_tensor=hidden_state,
        cache_position=pos,
        past_key_values=kv_cache,
        output_attentions=False,
    )
    pos_embed = model.model.model.rotary_emb(hidden_state, pos[None, :])

    for layer in model.model.model.layers:
        hidden_state = layer(
            hidden_state,
            attention_mask=causal_mask.contiguous()
            if causal_mask is not None
            else None,
            position_ids=pos[None, :],
            cache_position=pos,
            past_key_value=kv_cache,
            output_attentions=False,
            use_cache=kv_cache is not None,
            position_embeddings=[i.contiguous() for i in pos_embed],
        )[0]

    return model.lm_head(model.model.model.norm(hidden_state))
